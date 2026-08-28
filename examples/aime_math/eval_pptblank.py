"""Score a prompt on the pptblank test split and report F1.

``eval_dataset.py`` reports mean accuracy, which is the wrong headline number for
this task: always answering "no" scores ~64% accuracy while being useless (F1 0).
This evaluator keeps per-example verdicts so it can report precision / recall / F1,
break them down per design style, and show how the ambiguous ``mild`` slides were
answered.

Two prompts are scored so the optimized one can be read against its baseline, the
same convention ``eval_dataset.py`` uses.

Usage::

    export DEEPSEEK_API_KEY=...
    python -m examples.aime_math.eval_pptblank \
        --baseline-file examples/aime_math/prompts/pptblank_seed.txt \
        --optimized-file /path/to/optimized.txt

Env: the usual ``AIME_DEEPSEEK_MODEL`` / ``DEEPSEEK_API_BASE`` / ``AIME_SOLVER_*``
knobs, plus ``AIME_EVAL_K`` to cap the split and ``AIME_EVAL_NUM_THREADS``.
"""

import argparse
import collections
import json
import os

import dspy

from examples.aime_math.scoring import parse_verdict
from examples.aime_math.utils import configure_solver_lm, load_math_dataset, run_llm


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def f1_from(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return precision, recall, f1


def evaluate_prompt(prompt: str, dataset, num_threads: int) -> list[dict]:
    """Run ``prompt`` over ``dataset``, returning one record per example.

    Uses a thread pool directly rather than ``dspy.Evaluate`` because we need the
    individual verdicts, not just their mean. A failed call (after dspy's own
    retries) is recorded as an unparseable verdict rather than aborting the sweep.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(example):
        try:
            prediction = run_llm(example, prompt)
            raw = getattr(prediction, "answer", "")
            error = None
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            raw, error = "", f"{type(e).__name__}: {e}"
        verdict, boxed = parse_verdict(raw)
        return {
            "id": example.id,
            "gold": example.answer,
            "style": example._style_en,
            "raw": str(raw)[:120],
            "verdict": verdict,
            "boxed": boxed,
            "error": error,
        }

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(pool.map(one, dataset))


def summarize(records: list[dict]) -> dict:
    """Confusion counts and F1, overall and per style.

    ``either``-gold slides are excluded from F1: they are correct by construction,
    so counting them would inflate TP with slides that cannot be gotten wrong.
    Their verdict distribution is reported separately as a calibration signal.
    """
    def _zero():
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unparseable": 0, "errors": 0, "unboxed": 0}

    overall = _zero()
    per_style: dict[str, dict] = collections.defaultdict(_zero)
    mild = collections.Counter()

    for r in records:
        buckets = (overall, per_style[r["style"]])
        if r["error"]:
            for b in buckets:
                b["errors"] += 1
        # Format compliance is tracked across every slide, mild included: it is a
        # property of the prompt, not of the label.
        if r["verdict"] is not None and not r.get("boxed"):
            for b in buckets:
                b["unboxed"] += 1
        if r["gold"] == "either":
            mild[r["verdict"] or "unparseable"] += 1
            continue
        if r["verdict"] is None:
            # No verdict is a wrong answer: a miss on a positive, a false alarm
            # on a negative. Silently dropping these would flatter the numbers.
            for b in buckets:
                b["unparseable"] += 1
                b["fn" if r["gold"] == "yes" else "fp"] += 1
            continue
        correct = r["verdict"] == r["gold"]
        for b in buckets:
            if r["gold"] == "yes":
                b["tp" if correct else "fn"] += 1
            else:
                b["tn" if correct else "fp"] += 1

    def finish(c: dict) -> dict:
        p, rec, f1 = f1_from(c["tp"], c["fp"], c["fn"])
        decisive = c["tp"] + c["fp"] + c["fn"] + c["tn"]
        return {
            **c, "precision": p, "recall": rec, "f1": f1, "decisive": decisive,
            "accuracy_decisive": (c["tp"] + c["tn"]) / decisive if decisive else 0.0,
        }

    return {
        "overall": finish(overall),
        "per_style": {k: finish(v) for k, v in sorted(per_style.items())},
        "mild_verdicts": dict(mild),
    }


def print_report(label: str, s: dict) -> None:
    o = s["overall"]
    print(f"\n=== {label} ===")
    print(
        f"  F1={o['f1']:.3f}  precision={o['precision']:.3f}  recall={o['recall']:.3f}   "
        f"TP={o['tp']} FP={o['fp']} FN={o['fn']} TN={o['tn']}  "
        f"(accuracy on decisive slides {o['accuracy_decisive']:.3f})"
    )
    if o["unparseable"]:
        print(f"  {o['unparseable']} unparseable verdict(s), counted as wrong")
    if o["unboxed"]:
        print(f"  {o['unboxed']} verdict(s) not wrapped in \\box{{}} (graded, but format drifted)")
    if o["errors"]:
        print(f"  {o['errors']} call error(s) after retries")

    P, N = o["tp"] + o["fn"], o["fp"] + o["tn"]
    if P:
        _, _, all_yes = f1_from(P, N, 0)
        print(f"  reference: all-yes F1={all_yes:.3f}, all-no F1=0.000 ({P} pos / {N} neg)")

    print("  per style:")
    for style, c in s["per_style"].items():
        note = "  (few positives — F1 is noisy)" if c["tp"] + c["fn"] < 10 else ""
        print(
            f"    {style:19s} F1={c['f1']:.3f} prec={c['precision']:.3f} rec={c['recall']:.3f} "
            f"({c['tp'] + c['fn']} pos / {c['fp'] + c['tn']} neg){note}"
        )
    if s["mild_verdicts"]:
        total = sum(s["mild_verdicts"].values())
        print(f"  mild slides (either answer correct, excluded from F1): {s['mild_verdicts']} of {total}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline-file", help="file holding the baseline prompt")
    ap.add_argument("--optimized-file", help="file holding the optimized prompt")
    ap.add_argument("--split", default=os.environ.get("AIME_EVAL_SPLIT", "test"),
                    choices=["train", "val", "test"])
    ap.add_argument("--output", default=os.environ.get("AIME_EVAL_OUTPUT"),
                    help="write per-example verdicts and summaries to this JSON path")
    args = ap.parse_args()

    default_seed = os.path.join(os.path.dirname(__file__), "prompts", "pptblank_seed.txt")
    baseline_path = args.baseline_file or default_seed
    with open(baseline_path, encoding="utf-8") as f:
        baseline_prompt = f.read().strip()
    optimized_prompt = None
    if args.optimized_file:
        with open(args.optimized_file, encoding="utf-8") as f:
            optimized_prompt = f.read().strip()

    os.environ.setdefault("AIME_DATASET", "pptblank")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.luminai.cc/v1")
    model = os.environ.get("AIME_DEEPSEEK_MODEL", "openai/gemini-3-flash")
    num_threads = _env_int("AIME_EVAL_NUM_THREADS", 8)

    cache_dir = os.environ.get("AIME_CACHE_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        dspy.configure_cache(disk_cache_dir=cache_dir)

    configure_solver_lm(
        model, api_key, api_base,
        max_tokens=_env_int("AIME_SOLVER_MAX_TOKENS", 2000),
        temperature=_env_float("AIME_SOLVER_TEMPERATURE", 0.0),
    )

    k = _env_int("AIME_EVAL_K", 0) or None
    sizes = {"train": (k, None, None), "val": (None, k, None), "test": (None, None, k)}[args.split]
    splits = dict(zip(("train", "val", "test"),
                      load_math_dataset(name="pptblank", sizes=sizes, seed=_env_int("AIME_SEED", 42)),
                      strict=True))
    dataset = splits[args.split]

    golds = collections.Counter(ex.answer for ex in dataset)
    print(f"[eval] pptblank/{args.split}  n={len(dataset)}  model={model}  threads={num_threads}")
    print(f"[eval] gold distribution: {dict(golds)}")

    results = {}
    baseline_records = evaluate_prompt(baseline_prompt, dataset, num_threads)
    results["baseline"] = summarize(baseline_records)
    print_report(f"baseline ({os.path.basename(baseline_path)})", results["baseline"])

    if optimized_prompt:
        optimized_records = evaluate_prompt(optimized_prompt, dataset, num_threads)
        results["optimized"] = summarize(optimized_records)
        print_report(f"optimized ({os.path.basename(args.optimized_file)})", results["optimized"])
        delta = results["optimized"]["overall"]["f1"] - results["baseline"]["overall"]["f1"]
        print(f"\n[eval] F1 delta = {delta:+.3f}")

    if args.output:
        payload = {
            "split": args.split, "n": len(dataset), "model": model,
            "baseline_prompt": baseline_prompt, "optimized_prompt": optimized_prompt,
            "summaries": results,
            "baseline_records": baseline_records,
            "optimized_records": optimized_records if optimized_prompt else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[eval] written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
