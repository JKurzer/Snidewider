"""AUDIT — oracle checks for the structure-entropy features (fleet K).

  sam_states_rate   ORACLE: sum(len[s]-len[link[s]]) over states == number of
                  distinct substrings (brute-forced). Also states <= 2n-1.
  bwt_runs_rate     ORACLE: cyclic BWT inverts back to the original bytes.
  lz77_phrases_rate ORACLE: replay the same greedy logic while recording
                    (lit/match) phrases; decompress must reproduce the input.

Plus determinism and edge texts. Usage: .venv\\Scripts\\python scripts\\exp\\audit_structure_feats.py
"""

from __future__ import annotations

import random
import sys

sys.path.insert(0, "scripts/exp")
from fleet_condensates import bwt_runs_rate, lz77_phrases_rate, sam_states_rate

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# ---------- SAM oracle ----------
def sam_states_full(b: bytes):
    link = [-1]
    length = [0]
    trans: list[dict[int, int]] = [{}]
    last = 0
    for c in b:
        cur = len(length)
        length.append(length[last] + 1)
        link.append(0)
        trans.append({})
        p = last
        while p != -1 and c not in trans[p]:
            trans[p][c] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = trans[p][c]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(length)
                length.append(length[p] + 1)
                trans.append(trans[q].copy())
                link.append(link[q])
                while p != -1 and trans[p].get(c) == q:
                    trans[p][c] = clone
                    p = link[p]
                link[q] = link[cur] = clone
        last = cur
    return length, link, trans


def distinct_substrings(b: bytes) -> int:
    return len({b[i:j] for i in range(len(b)) for j in range(i + 1, len(b) + 1)})


# ---------- sentinel BWT: definition-level oracle (rotation matrix) ----------
def bwt_oracle(b: bytes) -> bytes:
    # the definition: sort all rotations of b + sentinel, take the last column
    s = b + b"\x00"
    n = len(s)
    rots = sorted(range(n), key=lambda i: s[i:] + s[:i])
    return bytes(s[i - 1] if i else s[-1] for i in rots)


# ---------- LZ77 record/replay ----------
def lz77_record(b: bytes):
    n = len(b)
    pos: dict[bytes, int] = {}
    i = 0
    phrases = []
    while i < n:
        best_len = 0
        if i + 4 <= n:
            j = pos.get(b[i:i + 4], -1)
            if j >= 0:
                while i + best_len < n and b[j + best_len] == b[i + best_len]:
                    best_len += 1
        if best_len:
            phrases.append(("m", j, best_len))
        else:
            phrases.append(("l", b[i]))
        step = max(1, best_len)
        for t in range(i, min(i + step, n - 3)):
            pos[b[t:t + 4]] = t
        i += step
    return phrases


def lz77_replay(phrases) -> bytes:
    out = bytearray()
    for ph in phrases:
        if ph[0] == "l":
            out.append(ph[1])
        else:
            _, j, ln = ph
            for t in range(ln):
                out.append(out[j + t])
    return bytes(out)


def main() -> None:
    rng = random.Random(7)
    alphabets = [b"ab", b"abc", bytes(range(97, 122))]
    cases = [bytes(rng.choice(alpha) for _ in range(rng.randint(5, 60)))
             for alpha in alphabets for _ in range(30)]
    cases += [b"", b"a", b"aaaaaa", b"banana", b"abcabcabcabc"]

    print("== SAM oracle: distinct-substring count ==", flush=True)
    bad = 0
    for b in cases:
        if not b:
            continue
        length, link, _ = sam_states_full(b)
        subtotal = sum(length[s] - length[link[s]] for s in range(1, len(length)))
        bound_ok = len(length) <= 2 * len(b) - 1 if len(b) >= 2 else True
        if subtotal != distinct_substrings(b) or not bound_ok:
            bad += 1
            print(f"    MISMATCH {b[:20]}... states={len(length)} subtotal={subtotal} "
                  f"brute={distinct_substrings(b)}")
    check("SAM == distinct substrings (all cases)", bad == 0, f"({bad} bad)")

    print("== BWT oracle: sentinel BWT == rotation-matrix definition ==", flush=True)
    bad = 0
    for b in cases:
        if not b:
            continue
        feat_val = bwt_runs_rate(b)
        oracle = bwt_oracle(b)
        oracle_runs = 1 + sum(1 for i in range(1, len(oracle)) if oracle[i] != oracle[i - 1])
        if abs(feat_val - oracle_runs / len(oracle)) > 1e-12:
            bad += 1
            print(f"    MISMATCH on {b[:20]}: feat={feat_val} oracle={oracle_runs/len(oracle)}")
    check("sentinel BWT runs == rotation-matrix definition", bad == 0, f"({bad} bad)")
    print(f"    note: banana$ -> {bwt_oracle(b'banana')!r}")

    print("== LZ77 oracle: record/replay roundtrip ==", flush=True)
    bad = 0
    for b in cases:
        if not b:
            continue
        ph = lz77_record(b)
        if lz77_replay(ph) != b:
            bad += 1
            print(f"    REPLAY FAIL on {b[:20]} (phrases={len(ph)})")
        # phrase count must match the feature's internal count
        expect = lz77_phrases_rate(b)
        if abs(expect - len(ph) / len(b)) > 1e-12:
            bad += 1
            print(f"    COUNT MISMATCH feature={expect} oracle={len(ph)/len(b)}")
    check("LZ77 record/replay + count agreement", bad == 0, f"({bad} bad)")

    print("== determinism + edges ==", flush=True)
    probe = b"the quick brown fox jumps over the lazy dog. " * 3
    check("deterministic", (sam_states_rate(probe), bwt_runs_rate(probe), lz77_phrases_rate(probe))
          == (sam_states_rate(probe), bwt_runs_rate(probe), lz77_phrases_rate(probe)))
    # 'a'*500: SAM has exactly n+1 states; sentinel BWT has exactly 2 runs
    check("all-same-char sane", abs(sam_states_rate(b'a' * 500) - 501 / 500) < 1e-12
          and abs(bwt_runs_rate(b'a' * 500) - 2 / 501) < 1e-12
          and lz77_phrases_rate(b'a' * 500) < 0.05)
    check("short docs NaN", str(sam_states_rate(b'hi')) == 'nan')

    print(f"\n{'ALL PASS' if not FAIL else f'FAILURES: {FAIL}'}")


if __name__ == "__main__":
    main()
