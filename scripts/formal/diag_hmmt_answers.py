"""Diagnostic: can the latex scorer grade every HMMT gold answer?

For each split (train/val/test) we check two things per gold answer:
  1. self-score  -> score_latex(gold, gold) == 1.0   (must always hold)
  2. mv-parsable -> math_verify.parse(gold) yields a non-empty parse

A gold that self-scores 1.0 only via the string-equality fallback (i.e. NOT
mv-parsable) is fragile: a model giving a symbolically-equivalent-but-differently
formatted answer would be marked wrong. We surface those loudly.
"""

import sys

from examples.aime_math.datasets import load_splits
from examples.aime_math.scoring import _HAS_MATH_VERIFY, score_latex

if _HAS_MATH_VERIFY:
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify


def mv_parsable(ans: str) -> bool:
    if not _HAS_MATH_VERIFY:
        return False
    try:
        parsed = mv_parse(str(ans))
        return bool(parsed)  # empty list == parse failure
    except Exception:
        return False


def mv_self_verifies(ans: str) -> bool:
    if not _HAS_MATH_VERIFY:
        return False
    try:
        p = mv_parse(str(ans))
        return bool(p) and mv_verify(p, p)
    except Exception:
        return False


def check_split(name, records):
    print(f"\n=== {name}: {len(records)} records ===")
    self_fail, not_parsable, no_selfverify = [], [], []
    for i, r in enumerate(records):
        gold = r["answer"]
        s, _ = score_latex(gold, gold)
        if s != 1.0:
            self_fail.append((i, gold))
        if not mv_parsable(gold):
            not_parsable.append((i, gold))
        elif not mv_self_verifies(gold):
            no_selfverify.append((i, gold))

    print(f"  self-score==1.0 failures : {len(self_fail)}")
    for i, g in self_fail:
        print(f"      [{i}] gold={g!r}")
    print(f"  NOT math_verify-parsable : {len(not_parsable)}  "
          f"(these pass only by exact string match -> fragile)")
    for i, g in not_parsable:
        print(f"      [{i}] gold={g!r}")
    print(f"  parsable but self-verify fails : {len(no_selfverify)}")
    for i, g in no_selfverify:
        print(f"      [{i}] gold={g!r}")
    return len(self_fail), len(not_parsable)


def main():
    print(f"math_verify available: {_HAS_MATH_VERIFY}")
    train, val, test = load_splits("hmmt", seed=0)
    total_selffail = 0
    total_fragile = 0
    for name, recs in (("train", train), ("val", val), ("test", test)):
        sf, nf = check_split(name, recs)
        total_selffail += sf
        total_fragile += nf
    print("\n=== SUMMARY ===")
    print(f"  self-score failures (BROKEN grading): {total_selffail}")
    print(f"  fragile golds (string-match only)   : {total_fragile}")
    # Only a self-score failure is a hard bug; fragile golds are a warning.
    sys.exit(1 if total_selffail else 0)


if __name__ == "__main__":
    main()
