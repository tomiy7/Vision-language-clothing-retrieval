from typing import Sequence


def recall_at_k(ranks: Sequence[int], k: int) -> float:
    # ranks - position of the correct result for each query.
    # k - number of top results to consider.

    if not ranks:
        return 0.0

    return sum(rank <= k for rank in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    # ranks - position of the correct result for each query.

    if not ranks:
        return 0.0

    return sum(1 / rank for rank in ranks) / len(ranks)


def mean_rank(ranks: Sequence[int]) -> float:
    # ranks - position of the correct result for each query.

    if not ranks:
        return 0.0

    return sum(ranks) / len(ranks)