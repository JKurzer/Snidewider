"""dct_shapes tests: the D3-winning tail-shaped feature set."""

import math

from ai_text_detection import dct_shapes

TEXT = (
    "The committee released its findings on Tuesday after months of deliberation. "
    "Officials said the report would guide policy for years. Critics were unconvinced, "
    "calling the language vague and the timeline optimistic. Supporters pointed to the "
    "rare bipartisan vote. The markets shrugged. By Friday, the story had shifted again "
    "as new documents surfaced. What happens next depends on the Senate calendar."
) * 3


def test_nobase_set_is_50_features():
    feats = dct_shapes.dct_tail_features(TEXT)
    assert len(feats) == 50
    assert len(dct_shapes.dct_tail_vector(TEXT)) == 50


def test_deterministic_pure_function():
    assert dct_shapes.dct_tail_features(TEXT) == dct_shapes.dct_tail_features(TEXT)


def test_short_doc_yields_nans():
    feats = dct_shapes.dct_tail_features("One sentence only.")
    assert any(math.isnan(v) for v in feats.values())
