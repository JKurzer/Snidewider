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


from ai_text_detection.metrics import tpr_at_fpr


def test_tied_zero_mass_cannot_fake_fpr():
    # 99.9% humans score exactly 0.0: nearest-rank would put the bar at 0.0
    # and call every positive "FPR 1e-3". The achieved FPR must stay honest.
    humans = [0.0] * 999 + [1.0]
    ai = [0.5] * 100
    res = tpr_at_fpr(ai, humans, fpr=1e-3)
    assert res["fpr_achieved"] <= 1e-3
    assert res["tpr"] == 1.0  # all AI above the (now strictly-positive) bar


def test_continuous_scores_behave_normally():
    humans = [i / 1000 for i in range(1000)]  # max 0.999; fpr=0.01 -> bar ~0.989
    ai = [0.995] * 50 + [0.5] * 50
    res = tpr_at_fpr(ai, humans, fpr=0.01)
    assert res["fpr_achieved"] <= 0.01
    assert res["tpr"] == 0.5  # only the 0.995 half clears the bar
