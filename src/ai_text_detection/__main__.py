"""aidt — human-facing CLI for the string-similarity toolkit.

    aidt ck2 A B                  CK2 score: 0.0 = identical, 1.0 = max different
    aidt qgram A B [-q 3]         q-gram distance and similarity (Ukkonen)
    aidt bag A B                  bag distance and similarity (Bartolini)
    aidt search PATTERN FILE      Hanada substring search; one hit per line

Strings are compared as UTF-8 bytes. FILE is read raw. Exit code is always 0
on success; bad arguments exit 2 with usage.
"""

import argparse
import sys
from pathlib import Path

from ai_text_detection import ck2, qgram


def _cmd_ck2(args: argparse.Namespace) -> None:
    a, b = args.a.encode("utf-8"), args.b.encode("utf-8")
    print(f"{ck2.similarity(a, b):.6f}")


def _cmd_qgram(args: argparse.Namespace) -> None:
    a, b = args.a.encode("utf-8"), args.b.encode("utf-8")
    d = qgram.distance(a, b, args.q)
    print(f"distance   {d}")
    print(f"similarity {qgram.similarity(a, b, args.q):.6f}")


def _cmd_bag(args: argparse.Namespace) -> None:
    a, b = args.a.encode("utf-8"), args.b.encode("utf-8")
    print(f"distance   {qgram.bag_distance(a, b)}")
    print(f"similarity {qgram.bag_similarity(a, b):.6f}")


def _cmd_search(args: argparse.Namespace) -> None:
    t = Path(args.file).read_bytes()
    p = args.pattern.encode("utf-8")
    hits = qgram.search(t, p, q=args.q, k=args.k)
    for start, end in hits:
        print(f"{start}\t{end}\t{t[start:end][:60]!r}")
    print(f"# {len(hits)} hits", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aidt", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ck2 = sub.add_parser("ck2", help="CK2 score (0=identical, 1=max different)")
    p_ck2.add_argument("a")
    p_ck2.add_argument("b")
    p_ck2.set_defaults(func=_cmd_ck2)

    p_qg = sub.add_parser("qgram", help="q-gram distance/similarity")
    p_qg.add_argument("a")
    p_qg.add_argument("b")
    p_qg.add_argument("-q", type=int, default=3)
    p_qg.set_defaults(func=_cmd_qgram)

    p_bag = sub.add_parser("bag", help="bag distance/similarity")
    p_bag.add_argument("a")
    p_bag.add_argument("b")
    p_bag.set_defaults(func=_cmd_bag)

    p_se = sub.add_parser("search", help="Hanada substring search of PATTERN in FILE")
    p_se.add_argument("pattern")
    p_se.add_argument("file")
    p_se.add_argument("-q", type=int, default=5)
    p_se.add_argument("-k", type=int, default=None, help="max distance (default: len(pattern)-q)")
    p_se.set_defaults(func=_cmd_search)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
