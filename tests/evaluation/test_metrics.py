from vision_language_clothing_retrieval.evaluation.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    mean_rank,
)


def test_recall_at_k():
    ranks = [1, 2, 1, 7, 12]

    assert recall_at_k(ranks, 1) == 0.4
    assert recall_at_k(ranks, 5) == 0.6
    assert recall_at_k(ranks, 10) == 0.8


def test_mean_reciprocal_rank():
    ranks = [1, 2, 4]

    expected = (1 / 1 + 1 / 2 + 1 / 4) / 3

    assert mean_reciprocal_rank(ranks) == expected


def test_mean_rank():
    ranks = [1, 2, 4]

    assert mean_rank(ranks) == 7 / 3


def test_empty_ranks():
    ranks = []

    assert recall_at_k(ranks, 1) == 0.0
    assert mean_reciprocal_rank(ranks) == 0.0
    assert mean_rank(ranks) == 0.0