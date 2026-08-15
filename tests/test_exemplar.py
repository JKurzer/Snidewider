"""Exemplar-bank features: synthetic oracle checks."""

from ai_text_detection import qgram
from ai_text_detection.exemplar import (
    EXEMPLAR_FEATURE_NAMES,
    ExemplarBank,
    exemplar_vector,
    profile_total,
)

AI_DOC = "the quick brown fox jumps over the lazy dog " * 8
HU_DOC = "qhxtv zkrjw mvbpd fglyw csnah ueoix rpqtk djzbw " * 8


def _prof(s: str, q: int = 3):
    return qgram.profile(s.encode("utf-8"), q)


def _banks():
    ai = ExemplarBank.from_texts([AI_DOC, "the lazy dog naps under the warm sun " * 8])
    hu = ExemplarBank.from_texts([HU_DOC, "wqvbx jkzrt plmnd fgycs hauio exqwk " * 8])
    return ai, hu


def test_profile_total_matches_doc_length():
    prof = _prof(AI_DOC)
    assert profile_total(prof) == len(AI_DOC.encode()) - 3 + 1


def test_identical_doc_has_zero_min_distance():
    ai, hu = _banks()
    feats = dict(zip(EXEMPLAR_FEATURE_NAMES, exemplar_vector(_prof(AI_DOC), ai, hu)))
    assert feats["ex_ai_min"] == 0.0  # AI_DOC is ai bank member 0
    assert feats["ex_hu_min"] > 0.5  # unrelated text stays far


def test_leave_one_out_excludes_self_match():
    ai, hu = _banks()
    feats = dict(
        zip(EXEMPLAR_FEATURE_NAMES, exemplar_vector(_prof(AI_DOC), ai, hu, ai_self_index=0))
    )
    assert feats["ex_ai_min"] > 0.0  # self-match removed
    assert feats["ex_ai_min"] < feats["ex_hu_min"]  # still closer to AI bank


def test_ai_like_doc_has_negative_contrast():
    ai, hu = _banks()
    feats = dict(
        zip(EXEMPLAR_FEATURE_NAMES, exemplar_vector(_prof(AI_DOC), ai, hu, ai_self_index=0))
    )
    assert feats["ex_contrast_mean"] < 0.0
    assert feats["ex_contrast_min"] < 0.0
    assert feats["ex_ai_mean_raw"] < feats["ex_hu_mean_raw"]


def test_normalized_distances_are_bounded():
    ai, hu = _banks()
    for doc in (AI_DOC, HU_DOC):
        feats = dict(zip(EXEMPLAR_FEATURE_NAMES, exemplar_vector(_prof(doc), ai, hu)))
        for name in ("ex_ai_min", "ex_ai_mean", "ex_ai_p10", "ex_hu_min", "ex_hu_mean", "ex_hu_p10"):
            assert 0.0 <= feats[name] <= 1.0
