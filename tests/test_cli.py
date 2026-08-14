"""CLI smoke tests: the aidt front door works end to end."""

import subprocess
import sys

CMD = [sys.executable, "-m", "ai_text_detection"]


def run(*args: str) -> subprocess.CompletedProcess:
    result = subprocess.run([*CMD, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def test_ck2_identical_and_different():
    assert run("ck2", "kitten", "kitten").stdout.strip() == "0.000000"
    assert float(run("ck2", "kitten", "sitting").stdout) > 0.0


def test_qgram_and_bag():
    out = run("qgram", "abc", "abd").stdout
    assert "distance   2" in out
    out = run("bag", "aab", "abb").stdout
    assert "distance   1" in out


def test_search(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_bytes(b"the quick brown fox jumps over the lazy dog")
    out = run("search", "brown fox", str(doc), "-q", "2", "-k", "0").stdout
    start = doc.read_bytes().index(b"brown fox")
    assert out.startswith(f"{start}\t{start + 9}\t")
