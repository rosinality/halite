import hashlib


def _fingerprint(*args) -> int:
    """A 128-bit fingerprint based on md5.

    For data shuffling - not for cryptography.

    Args:
        *args: any argument list that can be converted to a string

    Returns:
        an integer in [0, 2 ** 128)
    """
    return int.from_bytes(hashlib.md5(str(args).encode()).digest(), "little")


def index_shuffle(index: int, max_index: int, seed: int, rounds: int) -> int:
    """computes the position of `index` after a pseudorandom permutation on `[0, max_index])`.

    Based on Feistel ciphers.

    For data shuffling - not for cryptography.

    if i != j, then
    pseudorandom_permutation(n, i, seed) != pseudorandom_permutation(n, j, seed)

    Args:
        index: an integer in [0, max_index)
        max_index: A positive integer.
        seed: A posivtive integer used as seed for the pseudorandom permutation.
        rounds: Ignored. For compatibility with C++ version.

    Returns:
        An integer in [0, max_index].
    """
    del rounds
    if not isinstance(max_index, int):
        raise ValueError("n must be an integer")

    if index < 0 or index > max_index:
        raise ValueError("out of range")

    if max_index == 1:
        return 0

    # smallest k such that max_index fits in 2k bits
    k = (max_index.bit_length() + 1) // 2
    assert max_index <= 4**k
    # Permute repeatedly in [max_index, 4 ** k) until you land back in
    # [0, max_index]. This constitutes a permutation of [0, max_index].
    while True:
        # Feistel ciper on 2k bits - i.e. a permutation of [0, 4 ** k)
        a, b = index // (2**k), index % (2**k)
        for r in range(3):
            a, b = b, a ^ (_fingerprint(b, r, seed) % (2**k))
        index = a * (2**k) + b
        if index <= max_index:
            return int(index)
