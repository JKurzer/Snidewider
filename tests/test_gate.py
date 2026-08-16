"""Gate tests: spec loads, invariants hold, behavior is sane."""

from ai_text_detection.gate import SPEC_PATH, Gate


def test_spec_loads_with_six_components():
    gate = Gate.load(SPEC_PATH)
    assert len(gate.components) == 6
    for comp in gate.components:
        assert {"run", "levels", "window", "samples", "mode", "stat", "direction", "threshold"} <= set(comp)


def test_flag_is_deterministic_bool():
    gate = Gate.load(SPEC_PATH)
    text = "The quick brown fox jumps over the lazy dog. " * 40
    first = gate.flag(text)
    assert isinstance(first, bool)
    assert first == gate.flag(text)
    assert set(gate.report(text)) == {c["name"] for c in gate.components}


def test_short_and_empty_docs_never_crash_or_fire():
    gate = Gate.load(SPEC_PATH)
    assert gate.flag("") is False
    assert gate.flag("tiny") is False
    assert gate.flag("one two three") is False


def test_uniform_rhythm_scores_above_varied():
    # component_score with direction=low: a mechanically uniform doc should
    # out-score a varied one on every rand/step rhythm component
    gate = Gate.load(SPEC_PATH)
    uniform = "ab cd ef gh " * 300
    varied = " ".join(f"w{i}x" for i in range(1200))
    comp = gate.components[0]
    from ai_text_detection.gate import component_score

    assert component_score(uniform, comp) > component_score(varied, comp)
