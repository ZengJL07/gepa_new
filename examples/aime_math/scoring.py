"""Pluggable answer scorers, keyed by ``answer_type``.


Different datasets grade answers differently: AIME answers are integers, MATH-500
answers are LaTeX strings needing symbolic equivalence, and a future code dataset
(e.g. mbpp) would run test cases. Each scorer is a function

    scorer(gold, predicted) -> (score: float, feedback: str)

registered in ``SCORERS``. To add a dataset with a new grading rule, write a
scorer and register it here — no change to the eval loop or main.py is needed.
"""

import re

try:
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify

    _HAS_MATH_VERIFY = True
except ImportError:  # pragma: no cover - exercised only in envs without the extra
    _HAS_MATH_VERIFY = False


def score_int(gold, predicted) -> tuple[float, str]:
    """Exact integer match — AIME's original grading rule.

    Returns 0 with a parse-failure message if the prediction is not a valid
    integer (a reasoning model that ran out of tokens often lands here).
    """
    try:
        gold_int = int(gold)
    except (ValueError, TypeError):
        # gold should always be an int for this scorer; surface loudly.
        return 0.0, f"Dataset error: gold answer '{gold}' is not an integer."

    try:
        pred_int = int(predicted)
    except (ValueError, TypeError):
        return 0.0, (
            f"The final answer must be a valid integer and nothing else. You responded with "
            f"'{predicted}', which couldn't be parsed as a python integer. The correct answer is '{gold_int}'."
        )

    score = float(gold_int == pred_int)
    status = "correct" if score == 1.0 else "incorrect"
    return score, f"Your answer is {status}. The correct answer is '{gold_int}'."


def _normalize_latex(s: str) -> str:
    """Cheap normalization for the string-equality fallback."""
    s = str(s).strip()
    for token in ("\\left", "\\right", "$", " ", "\\,", "\\!"):
        s = s.replace(token, "")
    return s.lower()


def score_latex(gold, predicted) -> tuple[float, str]:
    """Symbolic equivalence for LaTeX/math-string answers (MATH-500).

    Uses ``math_verify`` (parse + verify) when available; falls back to a
    normalized string comparison otherwise (or when parsing fails, which
    ``math_verify`` does on some tuple/interval forms).
    """
    gold_s, pred_s = str(gold), str(predicted)

    if _HAS_MATH_VERIFY:
        try:
            gold_parsed = _mv_parse(gold_s)
            pred_parsed = _mv_parse(pred_s)
            # math_verify.verify(gold, target): True iff symbolically equal.
            if _mv_verify(gold_parsed, pred_parsed):
                return 1.0, f"Your answer is correct. The correct answer is '{gold_s}'."
        except Exception:
            pass  # fall through to string comparison

    if _normalize_latex(gold_s) == _normalize_latex(pred_s):
        return 1.0, f"Your answer is correct. The correct answer is '{gold_s}'."

    hint = "" if _HAS_MATH_VERIFY else " (install the 'math-verify' package for symbolic grading)"
    return 0.0, (
        f"Your answer is incorrect. You responded with '{pred_s}', but the correct answer is "
        f"'{gold_s}'.{hint} Provide the final answer in the same form as the reference."
    )


_YES_TOKENS = {"yes", "y", "true", "1", "是", "有", "有问题", "存在", "有过度留白"}
_NO_TOKENS = {"no", "n", "false", "0", "否", "没有", "无", "无问题", "不存在", "没有过度留白"}


def _token_to_verdict(text: str) -> str | None:
    """Map one already-isolated answer token onto "yes"/"no", else None."""
    s = text.strip().strip("。.,，、!！?？\"'`*$：: \n\t").lower()
    if s in _YES_TOKENS:
        return "yes"
    if s in _NO_TOKENS:
        return "no"
    return None


# ``\box{yes}`` as requested, plus ``\boxed{...}`` (the real LaTeX macro, which
# models emit by habit) and an optional ``$...$`` math wrapper. Case-insensitive
# and whitespace-tolerant, because none of that changes the verdict.
_BOX_RE = re.compile(r"\\box(?:ed)?\s*\{\s*([^{}]*?)\s*\}", re.IGNORECASE)


def extract_boxed(predicted) -> str | None:
    """Return the content of the LAST ``\\box{...}`` / ``\\boxed{...}`` in the text.

    Last rather than first: reasoning that mentions the format before committing
    ("I will answer \\box{yes} or \\box{no}... \\box{no}") must be read as its
    final answer, matching how boxed-answer conventions work elsewhere.
    """
    if predicted is None:
        return None
    matches = _BOX_RE.findall(str(predicted))
    return matches[-1] if matches else None


def parse_verdict(predicted) -> tuple[str | None, bool]:
    """Extract the yes/no verdict from a model answer.

    Returns ``(verdict, boxed)`` where ``verdict`` is ``"yes"``/``"no"``/None and
    ``boxed`` says whether it came from the requested ``\\box{...}`` wrapper.

    The box is the requested format and is tried first. A bare verdict is still
    accepted so a formatting slip does not get scored as a wrong *answer* — but
    ``boxed=False`` is reported back so the reflection LM sees the format drift and
    can tighten the prompt.

    Without a box, a leading Latin verdict counts only when punctuation follows it:
    a plain prefix test would read "none", "not sure" and "nothing stands out" as a
    confident "no", silently turning a non-answer into a graded verdict (and, on a
    negative gold, a spuriously *correct* one).
    """
    boxed_content = extract_boxed(predicted)
    if boxed_content is not None:
        verdict = _token_to_verdict(boxed_content)
        if verdict is not None:
            return verdict, True
        # A box holding something that is not a verdict ("\box{maybe}") is a
        # genuine non-answer; do not fall through and scavenge the prose around it.
        return None, True

    if predicted is None:
        return None, False
    raw = str(predicted)
    verdict = _token_to_verdict(raw)
    if verdict is not None:
        return verdict, False

    s = raw.strip().strip("。.,，、!！?？\"'`*：: \n\t").lower()
    match = re.match(r"(yes|no)\s*[,.;:—–-]", s)
    if match:
        return match.group(1), False
    for token, label in (("是", "yes"), ("否", "no"), ("有过度留白", "yes"),
                         ("没有", "no"), ("无过度留白", "no")):
        if s.startswith(token):
            return label, False
    return None, False


def _normalize_yesno(predicted) -> str | None:
    """Verdict only, discarding the box-format signal. Kept for callers that
    just need the answer."""
    return parse_verdict(predicted)[0]


def score_yesno(gold, predicted) -> tuple[float, str]:
    """Binary verdict grading with a deliberately unscorable third gold state.

    ``gold`` is one of:

    * ``"yes"``    — the slide has a ``severe`` whitespace region; only "yes" is right.
    * ``"no"``     — the slide has no annotated region; only "no" is right.
    * ``"either"`` — the slide's regions are all ``mild``. The labelling spec says
      mild whitespace may reasonably be called a problem or not, so both answers
      score 1.0. Such slides are dropped from train/val by the preparation script
      (a constant 1.0 is not a learning signal) and kept in test for reporting.

    False negatives and false positives get *different* feedback. This is the only
    place F1's asymmetry can enter the optimization loop: GEPA aggregates
    per-example scores by averaging, so the score itself cannot express "a miss
    costs more than a false alarm" — the reflection text has to say it.
    """
    if gold == "either":
        return 1.0, (
            "This slide's whitespace regions are all mild severity, where either verdict "
            "is acceptable. Scored correct regardless of the answer."
        )

    if gold not in ("yes", "no"):
        return 0.0, f"Dataset error: gold label '{gold}' is not one of yes/no/either."

    norm, boxed = parse_verdict(predicted)
    if norm is None:
        return 0.0, (
            f"The final answer must be a single word wrapped in \\box{{}} — \\box{{yes}} or "
            f"\\box{{no}}. You responded with '{predicted}', which contains no recognizable "
            "verdict."
        )

    # A verdict that skipped the wrapper is graded on its merits, but the format
    # drift is reported so the reflection LM can pin the output format down.
    format_note = (
        ""
        if boxed
        else " Note: the verdict was not wrapped in \\box{}. State the final answer as "
        "\\box{yes} or \\box{no} so it can be parsed reliably."
    )

    if norm == gold:
        return 1.0, f"Your verdict is correct. The reference answer is '{gold}'.{format_note}"

    if gold == "yes":
        return 0.0, (
            "MISS (false negative): this slide contains a severe excessive-whitespace region, "
            "but the verdict was 'no'. Misses are penalized more heavily than false alarms by "
            "the F1 objective, so when an empty region is genuinely ambiguous, prefer 'yes'."
            + format_note
        )
    return 0.0, (
        "FALSE ALARM (false positive): this slide has no excessive-whitespace region, but the "
        "verdict was 'yes'. Normal page margins and deliberate decorative spacing are not "
        "defects — only gaps that break the slide's own spacing rhythm are." + format_note
    )


SCORERS = {
    "int": score_int,
    "latex": score_latex,
    "yesno": score_yesno,
}


def get_scorer(answer_type: str):
    """Look up a scorer by ``answer_type``; defaults to integer (AIME) grading."""
    if answer_type not in SCORERS:
        raise ValueError(f"Unknown answer_type={answer_type!r}; registered: {sorted(SCORERS)}.")
    return SCORERS[answer_type]
