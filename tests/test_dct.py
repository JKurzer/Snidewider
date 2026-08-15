"""DCT encoder tests: paper properties + plumbing. Synthetic embeddings only."""

import math

import numpy as np
import pytest

from ai_text_detection import dct


@pytest.fixture(autouse=True)
def synthetic_embeddings():
    """Inject a tiny deterministic embedding table instead of the npz."""
    rng = np.random.RandomState(7)
    vocab = {w: i for i, w in enumerate(["the", "cat", "sat", "on", "a", "mat", "dog", "ran"])}
    matrix = rng.rand(len(vocab), 8).astype(np.float32)
    old_vocab, old_matrix = dct._VOCAB, dct._MATRIX
    dct._VOCAB, dct._MATRIX = vocab, matrix
    yield
    dct._VOCAB, dct._MATRIX = old_vocab, old_matrix


def test_c0_is_proportional_to_mean():
    vecs = dct.embed_sentence("the cat sat on a mat")
    coeffs = dct.dct_coefficients(vecs, k=2)
    n = vecs.shape[0]
    expected = math.sqrt(2.0 / n) * vecs.sum(axis=0)  # cos(0) = 1 for k=0 row
    np.testing.assert_allclose(coeffs[0], expected, rtol=1e-5)


def test_shapes_and_flatten():
    vecs = dct.embed_sentence("the cat sat")
    assert dct.dct_coefficients(vecs, k=2).shape == (2, 8)
    assert dct.sentence_vector("the cat sat", k=2).shape == (16,)


def test_oov_sentence_yields_zeros():
    assert dct.embed_sentence("zzz qqq").shape == (0, 8)
    assert np.all(dct.dct_coefficients(dct.embed_sentence("zzz qqq")) == 0)


def test_determinism_and_feature_keys():
    text = "The cat sat on a mat. A dog ran fast. The mat was warm."
    first = dct.dct_features(text)
    assert first == dct.dct_features(text)
    assert set(first) == set(dct.DCT_FEATURE_NAMES)


def test_single_sentence_doc_is_nan():
    feats = dct.dct_features("The cat sat.")
    assert math.isnan(feats["dct_adjacent_mean"])
