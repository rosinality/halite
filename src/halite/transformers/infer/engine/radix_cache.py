from collections import defaultdict
import heapq

import torch


class Node:
    def __init__(self, parent=None, key=None, value=None):
        self.childs = defaultdict(Node)
        self.parent = parent
        self.key = key
        self.value = value
        self.lock_ref = 0

    def set(self, key, value):
        self.key = key
        self.value = value

    def add_child(self, key, node):
        self.childs[key] = node


def match_key(key1, key2):
    i = 0

    for k1, k2 in zip(key1, key2):
        if k1 != k2:
            break

        i += 1

    return i


class RadixCache:
    def __init__(self, request_to_token_pool, kv_pool):
        self.request_to_token_pool = request_to_token_pool
        self.kv_pool = kv_pool

        self.initialized = False
        self.initialize()

    def initialize(self):
        if self.initialized:
            return

        self.root = Node(key=[], value=[])
        self.root.lock_ref = 1
        self.evictable_size = 0

        self.initialized = True

    def cleanup(self):
        self.initialized = False
        self.initialize()

    def insert(self, key, value=None):
        if value is None:
            value = [val for val in key]

        return self._insert(self.root, key, value)

    def _insert(self, node, key, value):
        if len(key) == 0:
            return 0

        if key[0] in node.childs:
            child = node.childs[key[0]]
            prefix_len = match_key(child.key, key)

            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return prefix_len

                key = key[prefix_len:]
                value = value[prefix_len:]

                return prefix_len + self._insert(child, key, value)

            new_node = self.split_node(child.key, child, prefix_len)

            return prefix_len + self._insert(
                new_node, key[prefix_len:], value[prefix_len:]
            )

        if len(key):
            new_node = Node(node, key, value)
            node.add_child(key[0], new_node)
            self.evictable_size += len(value)

        return 0

    def cache_finished_request(self, request, token_ids=None):
        if token_ids is None:
            token_ids = (request.input_ids + request.output_ids)[:-1]

        kv_indices = self.request_to_token_pool.request_to_token[
            request.request_pool_id, : len(token_ids)
        ]

        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.kv_pool.free(kv_indices[len(request.prefix_ids) : new_prefix_len])

        self.request_to_token_pool.free(request.request_pool_id)
        self.decrease_lock_ref(request.last_node)

    def cache_unfinished_request(self, request, token_ids=None):
        if token_ids is None:
            token_ids = request.all_ids

        kv_ids = self.request_to_token_pool.request_to_token[
            request.request_pool_id, : len(token_ids)
        ]

        new_prefix_len = self.insert(token_ids, kv_ids.clone())
        self.kv_pool.free(kv_ids[len(request.prefix_ids) : new_prefix_len])

        new_ids, new_last_node = self.match(token_ids)
        assert len(new_ids) == len(token_ids)
        self.request_to_token_pool.write(
            (request.request_pool_id, slice(len(request.prefix_ids), len(new_ids))),
            new_ids[len(request.prefix_ids) :],
        )

        self.decrease_lock_ref(request.last_node)
        self.increase_lock_ref(new_last_node)
        request.prefix_ids = new_ids
        request.last_node = new_last_node

    def increase_lock_ref(self, node):
        delta = 0
        while node != self.root:
            if node.lock_ref == 0:
                self.evictable_size -= len(node.value)
                delta -= len(node.value)

            node.lock_ref += 1
            node = node.parent

        return delta

    def decrease_lock_ref(self, node):
        delta = 0
        while node != self.root:
            if node.lock_ref == 1:
                self.evictable_size += len(node.value)
                delta += len(node.value)

            node.lock_ref -= 1
            node = node.parent

        return delta

    def evict(self, n_tokens, evict_callback):
        leaves = self.collect_leaves()
        heapq.heapify(leaves)

        n_evicted = 0
        while n_evicted < n_tokens and len(leaves):
            leaf = heapq.heappop(leaves)

            if leaf == self.root:
                break

            if leaf.lock_ref > 0:
                continue

            evict_callback(leaf.value)
            n_evicted += len(leaf.value)
            self.delete_leaf(leaf)

            if len(leaf.parent.childs) == 0:
                heapq.heappush(leaves, leaf.parent)

    def collect_leaves(self):
        results = []
        stack = [self.root]

        while stack:
            current = stack.pop()

            if len(current.childs) == 0:
                results.append(current)

            else:
                stack.extend(current.childs.values())

        return results

    def delete_leaf(self, node):
        for k, v in node.parent.childs.items():
            if v == node:
                break

        del node.parent.childs[k]
        self.evictable_size -= len(node.key)

    def set_value(self, key, value):
        return self._set_value(self.root, key, value)

    def _set_value(self, node, key, value):
        if len(key) == 0:
            return

        if key[0] in node.childs:
            child = node.childs[key[0]]
            prefix_len = match_key(child.key, key)

            if prefix_len == len(child.key):
                child.value = value[:prefix_len]
                self._set_value(child, key[prefix_len:], value[prefix_len:])

    def match(self, key):
        value = []
        last_node = [self.root]

        self._match(self.root, key, value, last_node)

        if value:
            if not isinstance(value[0], torch.Tensor):
                value = sum(value, [])

            else:
                value = torch.cat(value)

        else:
            value = torch.tensor([], dtype=torch.int32)

        return value, last_node[0]

    def _match(self, node, key, value, last_node):
        if len(key) == 0:
            return

        if key[0] in node.childs:
            child = node.childs[key[0]]
            prefix_len = match_key(child.key, key)

            if prefix_len < len(child.key):
                new_node = self.split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node

            else:
                value.append(child.value)
                last_node[0] = child
                self._match(child, key[prefix_len:], value, last_node)

    def split_node(self, key, child, split_len):
        new_node = Node(child.parent, child.key[:split_len], child.value[:split_len])
        new_node.add_child(key[split_len], child)
        new_node.lock_ref = child.lock_ref
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.childs[key[0]] = new_node

        return new_node

    def print(self):
        self._print(self.root, 0)
        print(f"# tokens: {self.total_size()}")

    def _print(self, node, indent):
        for _, child in node.childs.items():
            print(" " * indent, len(child.key), child.key[:10], f"r={child.lock_ref}")
            self._print(child, indent + 2)

    def total_size(self):
        return self._total_size(self.root)

    def _total_size(self, node):
        result = len(node.value)

        for child in node.childs.values():
            result += self._total_size(child)

        return result

    def traverse_depth(self, max_depth=None):
        results = []
        stack = [(self.root, 0)]

        while len(stack) > 0:
            cur_node, depth = stack.pop()

            if 0 < depth and (max_depth is None or depth <= max_depth):
                results.append(cur_node)

            if max_depth is None or depth < max_depth:
                for child in cur_node.childs.values():
                    stack.append((child, depth + 1))

        return results

    def traverse(self):
        paths = []
        stack = [(self.root, [])]

        while stack:
            node, current_path = stack.pop()

            if len(node.childs) == 0:
                paths.append(current_path)

                continue

            for child in node.childs.values():
                next_path = current_path + [child.key]
                stack.append((child, next_path))

        return paths
