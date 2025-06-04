import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from halite.transformers.tokainfer.engine.monitoring import track_time_decorator


@dataclass
class PrefixTreeBlock:
    idx: int
    contents: tuple[int, ...] = field(default_factory=tuple)
    last_used_at: float = 0.0
    seq_ids: set[str] = field(default_factory=set)
    children: dict[tuple[int, ...], "PrefixTreeBlock"] = field(default_factory=dict)
    parent: Optional["PrefixTreeBlock"] = None

    def is_root(self):
        return self.idx == -1

    def detach_from_parent(self):
        assert self.parent is not None
        assert self.parent.children[self.contents] == self
        self.parent.children.pop(self.contents)
        self.parent = None

    def wipe(self):
        if self.parent is not None:
            self.detach_from_parent()
        self.children.clear()
        self.seq_ids.clear()
        self.contents = tuple()
        self.last_used_at = 0.0

    def __repr__(self):
        return f"Block(idx={self.idx}, contents={self.contents}, children={len(self.children)} used_by={len(self.seq_ids)})"

    def __hash__(self):
        return self.idx

    def __lt__(self, other: "PrefixTreeBlock"):
        return self.last_used_at < other.last_used_at

    def tree_repr(self):
        def indent(s: str, spacing: int):
            lines = s.split("\n")
            with_index = [" " * spacing + line for line in lines]
            return "\n".join(with_index)

        out_lines = [repr(self)]

        for child in self.children.values():
            out_lines.append(indent(child.tree_repr(), 2))

        return "\n".join(out_lines)


class NoSpaceException(ValueError):
    """Exception raised when there is insufficient space for allocating a block."""


def truncate_to_multiple(x: list, multiple: int):
    return x[: len(x) - (len(x) % multiple)]


def pick_blocks_for_allocation(
    num_blocks: int,
    available_floating: list[PrefixTreeBlock],
    available_leaves: set[PrefixTreeBlock],
    available_leaf_heap: list[PrefixTreeBlock] | None = None,
    allow_used_leaves_in_heap: bool = False,
):
    assert num_blocks > 0

    chosen_floating_blocks: list[PrefixTreeBlock] = []
    chosen_tree_blocks: list[PrefixTreeBlock] = []

    for _ in range(min(num_blocks, len(available_floating))):
        chosen_floating_blocks.append(available_floating.pop())

    if len(chosen_floating_blocks) == num_blocks:
        return chosen_floating_blocks, chosen_tree_blocks

    if available_leaf_heap is None:
        available_leaf_heap = list(available_leaves)
        heapq.heapify(available_leaf_heap)

    num_remaining = num_blocks - len(chosen_floating_blocks)

    def get_next_leaf():
        if not allow_used_leaves_in_heap:
            leaf = heapq.heappop(available_leaf_heap)
            available_leaves.remove(leaf)
            assert len(leaf.seq_ids) == 0
            return leaf
        else:
            while True:
                candidate_leaf = heapq.heappop(available_leaf_heap)
                is_free = len(candidate_leaf.seq_ids) == 0
                assert is_free == (candidate_leaf in available_leaves)
                if is_free:
                    available_leaves.remove(candidate_leaf)
                    return candidate_leaf

    for _ in range(num_remaining):
        leaf = get_next_leaf()
        assert len(leaf.children) == 0

        chosen_tree_blocks.append(leaf)

        parent = leaf.parent
        assert parent is not None

        leaf.wipe()

        if (
            len(parent.children) == 0
            and len(parent.seq_ids) == 0
            and not parent.is_root()
        ):
            heapq.heappush(available_leaf_heap, parent)
            available_leaves.add(parent)

    return chosen_floating_blocks, chosen_tree_blocks


class BlockAllocator:
    def __init__(self, num_blocks: int, page_size: int):
        self.num_blocks = num_blocks
        self.page_size = page_size

        self.initialize()

    def initialize(self):
        self.all_blocks = [PrefixTreeBlock(idx=i) for i in range(self.num_blocks)]
        self.floating_blocks = list(self.all_blocks)
        self.prefix_tree = PrefixTreeBlock(idx=-1)
        self.num_free_blocks = len(self.all_blocks)
        self.available_leaves = set()

    def cleanup(self):
        self.all_blocks = None
        self.floating_blocks = None
        self.prefix_tree = None
        self.num_free_blocks = None
        self.available_leaves = None

    def add_floating(self, block: PrefixTreeBlock):
        assert block.parent is None
        assert len(block.children) == 0
        assert len(block.seq_ids) == 0
        self.floating_blocks.append(block)

    def num_used_blocks(self):
        return len(self.all_blocks) - self.num_free_blocks

    def fraction_used(self):
        return 1 - (self.num_free_blocks / len(self.all_blocks))

    def fraction_floating(self):
        return len(self.floating_blocks) / len(self.all_blocks)

    def sanity_checks(self, seq_ids: set[str] | None = None):
        set_floating = set(self.floating_blocks)
        for block in self.all_blocks:
            if block not in set_floating:
                assert block.parent is not None or len(block.seq_ids) > 0

            if block.parent is not None:
                assert block.parent.children[block.contents] == block

            for child in block.children.values():
                assert child.parent == block

        for block in self.floating_blocks:
            assert block.parent is None
            assert len(block.children) == 0

        num_free_blocks = sum(1 for block in self.all_blocks if len(block.seq_ids) == 0)
        assert num_free_blocks == self.num_free_blocks

        # available leaf is 1) unused by a seq, 2) in the prefix tree, 3) has no children
        available_leaves = {
            block
            for block in self.all_blocks
            if len(block.seq_ids) == 0
            and block.parent is not None
            and len(block.children) == 0
        }
        assert (
            available_leaves == self.available_leaves
        ), f"{len(available_leaves)} {len(self.available_leaves)}"

        if seq_ids is not None:
            all_used_seq_ids = set()
            for block in self.all_blocks:
                all_used_seq_ids.update(block.seq_ids)

            assert all_used_seq_ids == seq_ids

    def prefix_match(self, input_ids: list[int]) -> list[PrefixTreeBlock]:
        cached_blocks = []
        cur_block_in_tree = self.prefix_tree

        # we must re-process (i.e. can't cache hit on) the last token in the prompt,
        # since we need the logits from this token for sampling the first completion token
        cacheable_input_ids = input_ids[:-1]

        for start in range(0, len(cacheable_input_ids), self.page_size):
            # can only cache full pages (for now)
            if start + self.page_size > len(cacheable_input_ids):
                break

            sliced_ids = tuple(cacheable_input_ids[start : start + self.page_size])

            if (existing_child := cur_block_in_tree.children.get(sliced_ids)) is None:
                break
            else:
                cached_blocks.append(existing_child)
                cur_block_in_tree = existing_child

        return cached_blocks

    def update_prefix_tree(self, blocks: list[PrefixTreeBlock], ids: list[int]):
        assert len(blocks) * self.page_size == len(ids)

        cur_block_in_tree = self.prefix_tree
        for i, block in enumerate(blocks):
            start = i * self.page_size
            page_ids = tuple(ids[start : start + self.page_size])

            if (existing_child := cur_block_in_tree.children.get(page_ids)) is not None:
                cur_block_in_tree = existing_child
            else:
                block.contents = page_ids
                block.parent = cur_block_in_tree
                cur_block_in_tree.children[page_ids] = block

                if cur_block_in_tree in self.available_leaves:
                    self.available_leaves.remove(cur_block_in_tree)

                if len(block.seq_ids) == 0:
                    self.available_leaves.add(block)

                cur_block_in_tree = block

    def assign_block_to_seq(self, block: PrefixTreeBlock, seq_id: str):
        assert seq_id not in block.seq_ids, f"{seq_id} {block}"

        if len(block.seq_ids) == 0:
            self.num_free_blocks -= 1

        block.seq_ids.add(seq_id)
        block.last_used_at = time.time()

        if block in self.available_leaves:
            self.available_leaves.remove(block)

    def num_blocks_needed(self, kv_indices: list[int], length: int):
        num_existing_blocks = len(kv_indices)
        num_existing_tokens = num_existing_blocks * self.page_size

        num_blocks_needed = max(
            0, math.ceil((length - num_existing_tokens) / self.page_size)
        )

        return num_blocks_needed

    @track_time_decorator()
    def allocate_up_to_length(
        self,
        seq_id: str,
        kv_indices: list[int],
        length: int,
        available_leaf_heap: list[PrefixTreeBlock] | None = None,
        allow_used_leaves_in_heap: bool = False,
    ):
        num_blocks_needed = self.num_blocks_needed(kv_indices, length)

        new_kv_indices: list[int] = []

        if num_blocks_needed == 0:
            return new_kv_indices

        if num_blocks_needed > self.num_free_blocks:
            raise NoSpaceException()

        floating_blocks, tree_blocks = pick_blocks_for_allocation(
            num_blocks=num_blocks_needed,
            available_floating=self.floating_blocks,
            available_leaves=self.available_leaves,
            available_leaf_heap=available_leaf_heap,
            allow_used_leaves_in_heap=allow_used_leaves_in_heap,
        )
        all_blocks = floating_blocks + tree_blocks

        for block in all_blocks:
            self.assign_block_to_seq(block, seq_id)
            new_kv_indices.append(block.idx)

        return new_kv_indices

    @track_time_decorator()
    def make_available_leaf_heap(self):
        """
        Valid to reuse across multiple consecutive calls to allocate_up_to_length.

        INVALIDATED by calls to allocate_with_prefix_match or free_and_update.
        """
        available_leaf_heap = list(self.available_leaves)
        heapq.heapify(available_leaf_heap)
        return available_leaf_heap

    def enough_free_blocks_for_allocation(
        self,
        num_existing_blocks: int,
        target_token_length: int,
        num_reserved_blocks: int,
    ):
        num_existing_tokens = num_existing_blocks * self.page_size

        num_blocks_needed = math.ceil(
            (target_token_length - num_existing_tokens) / self.page_size
        )

        return num_blocks_needed + num_reserved_blocks <= self.num_free_blocks

    @track_time_decorator()
    def allocate_with_prefix_match(
        self,
        seq_id: str,
        prompt_ids: list[int],
        num_reserved_blocks: int = 0,
        available_leaf_heap: list[PrefixTreeBlock] | None = None,
        allow_used_leaves_in_heap: bool = False,
    ):
        cached_blocks = self.prefix_match(prompt_ids)

        # when determining if we have enough free blocks to allocate this seq,
        # we need to account for any blocks in the prefix tree that were previously
        # unused but are now "soft-assigned" to the current seq (soft in the sense
        # that we'll only actually assign them if we can find enough remaining blocks).
        num_unused_cache_blocks = sum(
            1 for block in cached_blocks if len(block.seq_ids) == 0
        )

        num_cached_prompt_ids = len(cached_blocks) * self.page_size

        if not self.enough_free_blocks_for_allocation(
            num_existing_blocks=len(cached_blocks),
            target_token_length=len(prompt_ids),
            num_reserved_blocks=num_reserved_blocks + num_unused_cache_blocks,
        ):
            raise NoSpaceException()

        cached_kv_indices = [block.idx for block in cached_blocks]

        # important to do assignment before requesting remaining blocks
        # otherwise some of these cached blocks might be reassigned
        for block in cached_blocks:
            self.assign_block_to_seq(block, seq_id)

        try:
            allocated_kv_indices = self.allocate_up_to_length(
                seq_id,
                cached_kv_indices,
                len(prompt_ids),
                available_leaf_heap=available_leaf_heap,
                allow_used_leaves_in_heap=allow_used_leaves_in_heap,
            )
        except NoSpaceException:
            # if we don't have enough blocks, we should have raised a NoSpaceException
            # above - now we're in a state where we already
            # assigned some blocks to this seq
            raise RuntimeError("Shouldn't happen")

        full_kv_indices = cached_kv_indices + allocated_kv_indices
        assert len(full_kv_indices) == math.ceil(len(prompt_ids) / self.page_size)

        full_kv_blocks = [self.all_blocks[idx] for idx in full_kv_indices]

        cachable_prompt_ids = truncate_to_multiple(prompt_ids, self.page_size)
        num_cacheable_blocks = len(cachable_prompt_ids) // self.page_size

        self.update_prefix_tree(
            full_kv_blocks[:num_cacheable_blocks], cachable_prompt_ids
        )

        return full_kv_indices, num_cached_prompt_ids

    @track_time_decorator()
    def free_and_update(self, seq_id: str, kv_indices: list[int], token_ids: list[int]):
        allocated_blocks = [self.all_blocks[idx] for idx in kv_indices]

        newly_free_block_indices: list[int] = []

        # free the blocks - reversed order matters so we
        # can move the tail over, described below
        for block in reversed(allocated_blocks):
            block.seq_ids.remove(seq_id)
            if len(block.seq_ids) == 0:
                self.num_free_blocks += 1
                newly_free_block_indices.append(block.idx)

                # moving the "tail" in the prefix tree that was only used
                # by this sequence, as well as blocks that aren't in the
                # prefix tree (i.e. completion blocks), out of the tree
                if len(block.children) == 0:
                    block.wipe()

        cachable_full_ids = truncate_to_multiple(token_ids, self.page_size)

        num_cacheable_blocks = len(cachable_full_ids) // self.page_size
        cacheable_blocks = allocated_blocks[:num_cacheable_blocks]

        self.update_prefix_tree(cacheable_blocks, cachable_full_ids)

        # any block we freedthat's not in the prefix tree is now floating
        for block in allocated_blocks:
            if block.parent is None:
                self.add_floating(block)

        return newly_free_block_indices


class BatchIndexAllocator:
    def __init__(self, max_indices: int):
        self.max_indices = max_indices

        self.initialize()

    def initialize(self):
        self.available_indices = deque(range(self.max_indices))

    def cleanup(self):
        self.available_indices = None

    def allocate(self):
        out = self.available_indices.popleft()
        return out

    def free(self, idx: int):
        self.available_indices.append(idx)
