"""Metrics tests: tie handling in the rank statistic (F3 found this the hard
way — tuple-order tie-breaking fabricated AUROC 0.996 on discrete features)."""

from ai_text_detection.metrics import auroc


def test_perfect_separation():
    assert auroc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert auroc([0.0, 1.0], [2.0, 3.0]) == 0.0


def test_all_tied_is_chance():
    assert auroc([1.0, 1.0, 1.0], [1.0, 1.0]) == 0.5


def test_partial_ties_average_ranks():
    # pairs: (1>0)=1, (1==1)=0.5, (2>0)=1, (2>1)=1 -> 3.5/4
    assert auroc([1.0, 2.0], [0.0, 1.0]) == 0.875


def test_continuous_unchanged_by_tie_logic():
    assert auroc([0.9, 0.8, 0.7], [0.4, 0.3, 0.2]) == 1.0
