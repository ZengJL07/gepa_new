"""Synthetic, offline task for the tool-feedback loop (no network).

Task: a hidden integer ``target`` lives in ``[lo, hi]``. The model must find it by
repeatedly calling the ``probe`` tool, which reports "higher"/"lower"/"correct"
(classic number guessing — a binary search solves it in ~log2(range) turns).
When confident, the model submits ``<final>N</final>``.

This is a placeholder to exercise the whole GEPA + custom-sampling pipeline end
to end; ``probe`` and the example schema are swappable for a real task later.
"""

import random
from dataclasses import dataclass

from examples.tool_loop.tools import register_tool


@dataclass
class GuessExample:
    """One task instance. ``input`` is the prompt shown to the model."""

    input: str
    lo: int
    hi: int
    target: int
    # dspy-style hook so downstream code can treat it like other examples.

    def with_inputs(self, *_names):
        return self


@register_tool("probe", 'Probe an integer: {"value": N} -> "higher"/"lower"/"correct" relative to the target.')
def _probe(args: dict, example: GuessExample) -> str:
    """Local feedback: compare the probed value against the hidden target."""
    if "value" not in args:
        return 'Missing required arg "value" (an integer). Example: <call name="probe">{"value": 50}</call>.'
    try:
        guess = int(args["value"])
    except (ValueError, TypeError):
        return f'"value" must be an integer, got {args["value"]!r}.'

    if guess < example.lo or guess > example.hi:
        return f"{guess} is outside the allowed range [{example.lo}, {example.hi}]."
    if guess == example.target:
        return f"{guess} is correct! Submit it with <final>{guess}</final>."
    direction = "higher" if guess < example.target else "lower"
    return f"The target is {direction} than {guess}. Keep searching within [{example.lo}, {example.hi}]."


def _make_prompt(lo: int, hi: int) -> str:
    return (
        f"I am thinking of a secret integer between {lo} and {hi} (inclusive). "
        "Find it by probing values and reading the feedback, then submit the answer."
    )


def make_dataset(n: int, lo: int = 1, hi: int = 100, seed: int = 0) -> list[GuessExample]:
    """Generate ``n`` guessing tasks with targets drawn reproducibly from [lo, hi]."""
    rng = random.Random(seed)
    return [
        GuessExample(input=_make_prompt(lo, hi), lo=lo, hi=hi, target=rng.randint(lo, hi)) for _ in range(n)
    ]


def load_splits(
    train_n: int = 12, val_n: int = 8, test_n: int = 8, lo: int = 1, hi: int = 100, seed: int = 0
) -> tuple[list[GuessExample], list[GuessExample], list[GuessExample]]:
    """Three disjoint splits with independent seeds (still fully reproducible)."""
    train = make_dataset(train_n, lo, hi, seed=seed)
    val = make_dataset(val_n, lo, hi, seed=seed + 1)
    test = make_dataset(test_n, lo, hi, seed=seed + 2)
    return train, val, test
