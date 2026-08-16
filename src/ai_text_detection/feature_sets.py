"""Named feature sets from the fleet experiments, as reusable pure functions.

  relative_vector — F1's length-relative window set (full coverage)
  qgram12_vector  — F3's recommended 12-feature q-gram set
"""

from __future__ import annotations

from ai_text_detection import burst, qgram
from ai_text_detection.features import qgram_profile_features
from ai_text_detection.features_relative import document_features_relative

QGRAM12_NAMES = (
    "short_ck2_mean", "short_ck2_stdev", "mid_ck2_mean", "mid_ck2_stdev",
    "mid_qgram_mean", "mid_qgram_stdev", "q2_entropy", "q3_entropy",
    "q3_distinct_ratio", "q3_repeat_frac", "q3_max_share", "q5_top10_share",
)


def relative_vector(text: str) -> list[float]:
    feats = document_features_relative(text)
    return [feats[k] for k in sorted(feats)]


def qgram12_vector(text: str) -> list[float]:
    short = burst.burst_features(text, window=64, gap=0, unit="bytes")
    mid_ck2 = burst.burst_features(
        text, window=150, samples=32, min_gap=50, unit="tokens", mode="random"
    )
    mid_qg = burst.burst_features(
        text, window=150, samples=32, min_gap=50, unit="tokens", mode="random", metric="qgram"
    )
    q2 = qgram_profile_features(text, 2)
    q3 = qgram_profile_features(text, 3)
    prof5 = qgram.profile(text.encode("utf-8"), 5)
    total5 = sum(c for _, c in prof5)
    top10_share = sum(c for _, c in prof5[:10]) / total5 if total5 else float("nan")
    return [
        short["mean"], short["stdev"], mid_ck2["mean"], mid_ck2["stdev"],
        mid_qg["mean"], mid_qg["stdev"], q2["qgram_entropy"], q3["qgram_entropy"],
        q3["qgram_distinct_ratio"], q3["qgram_repeat_frac"], q3["qgram_max_share"],
        top10_share,
    ]
