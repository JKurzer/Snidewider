"""Hits feature tests: known-fragment detection, no-match case, purity."""

from ai_text_detection.hits import ChunkBank, hit_features

AI_DOC = "the quick brown fox jumps over the lazy dog and runs into the forest " * 8
HU_DOC = "my grandmother always said that patience is the quiet cousin of wisdom " * 8


def _banks():
    ai_bank = ChunkBank.from_texts([AI_DOC], chunk=150)
    hu_bank = ChunkBank.from_texts([HU_DOC], chunk=150)
    return ai_bank, hu_bank


def test_doc_containing_ai_fragment_hits_ai_bank():
    ai_bank, hu_bank = _banks()
    feats = hit_features(b"prefix " + AI_DOC.encode("utf-8") + b" suffix", ai_bank, hu_bank)
    assert feats["hits_ai_rate"] == 1.0
    assert feats["hits_hu_rate"] == 0.0
    assert feats["hits_contrast"] == 1.0


def test_unrelated_doc_hits_nothing():
    ai_bank, hu_bank = _banks()
    feats = hit_features(b"zyxwvutsrqponmlkjihgfedcba " * 30, ai_bank, hu_bank)
    assert feats["hits_ai_rate"] == 0.0
    assert feats["hits_hu_rate"] == 0.0


def test_pure_given_fixed_banks():
    ai_bank, hu_bank = _banks()
    doc = b"some sample document " * 20
    assert hit_features(doc, ai_bank, hu_bank) == hit_features(doc, ai_bank, hu_bank)
