"""Shape mapping tests: exact skeleton, run-map invariants, purity."""

from ai_text_detection import shape


def test_skeleton_map_exact():
    assert shape.skeleton_map("The cat sat.") == b"= = =."  # one = per word
    assert shape.skeleton_map("one  two\nthree!") == b"=  =\n=!"


def test_dct_run_map_symbols_and_length():
    text = " ".join(f"w{i}" for i in range(64))  # 64 tokens -> 8 runs
    stream = shape.dct_run_map(text)
    assert len(stream) == 8
    assert all(b >= 0 for b in stream)  # bytes by construction


def test_dct_run_map_distinguishes_rhythm():
    flat = shape.dct_run_map(" ".join(["same"] * 64))  # uniform lengths
    varied = shape.dct_run_map(" ".join(["a", "bb", "ccc", "dddd"] * 16))
    assert flat != varied


def test_shape_features_keys_and_purity():
    text = " ".join(f"token{i % 41}" for i in range(300))
    feats = shape.shape_features(text)
    assert set(feats) == set(shape.SHAPE_FEATURE_NAMES)
    assert feats == shape.shape_features(text)
