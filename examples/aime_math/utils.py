import os
import threading

import dspy

from examples.aime_math.datasets import get_spec, load_splits
from examples.aime_math.scoring import get_scorer
from gepa import Image

# dspy signals a max_tokens truncation only by logging a warning inside
# ``LM._check_truncation``. We want a truncated response (finish_reason ==
# "length") to score 0 even when a partial answer happened to parse, because the
# model likely ran away in its reasoning. ``_check_truncation`` runs on the same
# worker thread that then computes the metric, so a thread-local flag is a
# race-free way to carry that signal from the LM call into the evaluator.
_truncation_flag = threading.local()


class TruncationTrackingLM(dspy.LM):
    """dspy.LM that records, per-thread, whether the last response was truncated."""

    def _check_truncation(self, results):
        # Reset-then-set on every call so the flag always reflects THIS thread's
        # most recent LM call. dspy.Evaluate reuses worker threads, so a sticky
        # flag would wrongly taint every later sample on a thread that once saw
        # a truncation.
        _truncation_flag.hit = False
        try:
            if self.model_type != "responses" and any(c.finish_reason == "length" for c in results["choices"]):
                _truncation_flag.hit = True
        except Exception:
            pass
        return super()._check_truncation(results)


def truncation_hit() -> bool:
    """Whether the current thread's most recent LM call was truncated."""
    return getattr(_truncation_flag, "hit", False)


def reset_truncation_flag() -> None:
    _truncation_flag.hit = False


def configure_solver_lm(model: str, api_key: str, api_base: str, max_tokens: int, temperature: float = 1.0):
    """Build a truncation-tracking solver LM and set it as the active dspy LM.

    Shared by main.py (optimization) and eval_dataset.py (standalone eval) so
    both configure the solver identically. Returns the LM.
    """
    lm = TruncationTrackingLM(
        model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)
    return lm


class MathSolverSignature(dspy.Signature):
    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")


class SlideVerdictSignature(dspy.Signature):
    """Vision signature for the PPT excessive-whitespace task.

    The annotated type on ``input`` is what makes dspy serialize the value as an
    image content part rather than str()-ing it, so it must stay.
    """

    input: dspy.Image = dspy.InputField(desc="A rendered presentation slide.")
    answer = dspy.OutputField(desc="The verdict wrapped in \\box{}, e.g. \\box{yes} or \\box{no}.")


def _use_cot() -> bool:
    """Whether the solver wraps the signature in dspy.ChainOfThought.

    Controlled by AIME_SOLVER_USE_COT (default "1"). ChainOfThought injects an
    explicit ``reasoning`` output field the model must fill BEFORE ``answer`` —
    good for non-reasoning models, but redundant (and truncation-prone) for
    models that already reason internally. Set to "0"/"false" to use a bare
    dspy.Predict (``answer`` only).
    """
    return os.environ.get("AIME_SOLVER_USE_COT", "1").strip().lower() not in ("0", "false", "no")


def _active_input_type() -> str:
    """Input modality of the dataset this process is running, from AIME_DATASET.

    Read from the spec rather than its own env var so the modality can never
    disagree with the dataset being loaded.
    """
    return get_spec(os.environ.get("AIME_DATASET", "aime")).input_type


def _build_predictor():
    sig = SlideVerdictSignature if _active_input_type() == "image" else MathSolverSignature
    return dspy.ChainOfThought(sig) if _use_cot() else dspy.Predict(sig)


predictor = _build_predictor()


def image_root() -> str:
    """Directory that record-relative image paths resolve against."""
    return os.environ.get(
        "PPTBLANK_IMAGE_ROOT", os.path.join(os.path.dirname(__file__), "data", "pptblank")
    )


def _set_instructions(pred, prompt: str) -> None:
    """Set the candidate prompt as the signature instructions.

    ChainOfThought nests the real predictor at ``.predict``; Predict is itself
    the leaf and exposes ``.signature`` directly.
    """
    target = pred.predict if hasattr(pred, "predict") else pred
    target.signature.instructions = prompt


def run_llm(example, prompt: str):
    """Run the LLM on a single example with the given prompt."""
    _set_instructions(predictor, prompt)
    return predictor(input=example.input)


def math_metric(example, prediction):
    """Compute score and detailed feedback for a math problem.

    Grading is delegated to the scorer registered for the example's
    ``answer_type`` (attached by ``load_math_dataset``; defaults to ``"int"`` so
    plain AIME examples keep their integer-matching behaviour). The reference
    solution, when present, is appended to the feedback to help the optimizer.
    """
    # A response truncated at max_tokens (finish_reason == "length") is scored 0
    # even if a partial answer parsed: hitting the ceiling signals runaway
    # reasoning, so any surviving answer is unreliable.
    if truncation_hit():
        feedback_text = (
            "The model's response was truncated at the max_tokens limit — likely runaway "
            "reasoning — so its answer is unreliable and scored 0. Guide the model to reason "
            "concisely and end with a single final answer."
        )
        return 0.0, feedback_text

    answer_type = getattr(example, "answer_type", "int")
    scorer = get_scorer(answer_type)
    score, feedback_text = scorer(example.answer, getattr(prediction, "answer", None))

    written_solution = getattr(example, "solution", "")
    if written_solution:
        feedback_text += (
            f" Here's the full step-by-step solution:\n{written_solution}\n\n"
            "Think about what takeaways you can learn from this solution to improve your "
            "future answers and approach to similar problems"
        )
    return score, feedback_text


def _to_example(record: dict, answer_type: str, input_type: str = "text") -> dspy.Example:
    """Turn a dataset record (AIME-keyed dict) into a dspy.Example.

    Every key is carried onto the Example (original ``_``-prefixed fields
    included) and ``answer_type`` is attached so ``math_metric`` picks the right
    scorer. ``input`` is the sole model input.

    For ``input_type == "image"`` the record's ``input`` is a path relative to
    :func:`image_root` and is wrapped in a ``dspy.Image`` here — records stay
    plain strings on disk so the dataset cache remains JSON-serializable.
    """
    rec = dict(record)
    if input_type == "image":
        rec["input"] = dspy.Image(os.path.join(image_root(), rec["input"]))
    ex = dspy.Example(**rec, answer_type=answer_type)
    return ex.with_inputs("input")


def _describe_annotation(example) -> str:
    """Human-readable ground truth for the reflection prompt.

    Names each labelled region with its severity and canvas share so the
    reflection LM can connect the drawn boxes to a magnitude, and states the
    negative case explicitly — negatives are shown the *plain* slide (there is
    nothing to draw), and without this sentence the LM cannot tell "clean slide"
    from "annotations missing".
    """
    severities = getattr(example, "_severity", []) or []
    ratios = getattr(example, "_area_ratios", []) or []
    if not severities:
        return (
            "No labelled regions: this slide has NO excessive whitespace. "
            "Its empty areas are normal margins or intentional design spacing."
        )
    parts = ", ".join(
        f"{sev} ({ratio * 100:.1f}% of the canvas)"
        for sev, ratio in zip(severities, ratios, strict=False)
    )
    boxed = "box" if len(severities) == 1 else "boxes"
    return (
        f"{len(severities)} labelled excessive-whitespace region(s), drawn as coloured "
        f"{boxed} on the image (red = severe, orange = mild): {parts}."
    )


def build_side_info(example, prediction, score: float, feedback: str) -> dict:
    """Assemble the reflection payload for one evaluated example.

    Text datasets keep the original four keys. The image task instead sends the
    *annotated* slide via ``gepa.Image`` (only that wrapper is converted to an
    image content part by the reflection prompt builder — a ``dspy.Image`` would
    be str()-ed), plus the deck's design style, since layout convention differs
    per style and it is a generalizable axis; the topic is a brand name and is
    deliberately withheld to avoid overfitting to it.

    ``_``-prefixed keys are stripped from the reflection prompt by the adapter, so
    they carry bookkeeping only.
    """
    if getattr(example, "answer_type", "") != "yesno":
        return {
            "score": score,
            "input": example.input,
            "output": getattr(prediction, "answer", ""),
            "reasoning": getattr(prediction, "reasoning", ""),
            "execution_feedback": feedback,
        }

    annotated = os.path.join(image_root(), example._annotated_path)
    return {
        "score": score,
        "Slide": Image(path=annotated),
        "DesignStyle": example._style_en,
        "GroundTruthRegions": _describe_annotation(example),
        "ReferenceVerdict": example.answer,
        "ModelVerdict": getattr(prediction, "answer", ""),
        "reasoning": getattr(prediction, "reasoning", ""),
        "execution_feedback": feedback,
        "_image_id": example.id,
    }


def split_sizes_from_env():
    """Read per-split size caps from the environment.

    Returns ``(train_k, val_k, test_k)`` where each entry is an int cap or None.
    ``AIME_TRAIN_K`` / ``AIME_VAL_K`` / ``AIME_TEST_K`` set a split individually;
    each falls back to ``AIME_TRIM_K`` (applied to all splits), then 0 (uncapped).
    """
    default = int(os.environ.get("AIME_TRIM_K", 0) or 0)

    def _one(name: str) -> int | None:
        return int(os.environ.get(name, default) or 0) or None

    return (_one("AIME_TRAIN_K"), _one("AIME_VAL_K"), _one("AIME_TEST_K"))


def load_math_dataset(name: str = "aime", sizes=None, seed: int = 0):
    """Load a registered dataset as (trainset, valset, testset) of dspy.Examples.

    - ``name``: key into ``datasets.SPECS`` (default ``"aime"`` preserves the
      original two-source AIME behaviour).
    - ``sizes``: optional ``(train_k, val_k, test_k)`` tuple capping each split
      to its first k records; each entry may be None/0 to leave it uncapped.
    - ``seed``: shuffle seed for single-source splits (AIME ignores it to keep
      its historical ordering).
    """
    spec = get_spec(name)
    train, val, test = load_splits(name, seed=seed, sizes=sizes)
    trainset = [_to_example(r, spec.answer_type, spec.input_type) for r in train]
    valset = [_to_example(r, spec.answer_type, spec.input_type) for r in val]
    testset = [_to_example(r, spec.answer_type, spec.input_type) for r in test]
    return trainset, valset, testset


def evaluate_on_dataset(prompt, dataset):
    """Evaluate a predictor on a dataset using dspy.Evaluate."""
    _set_instructions(predictor, prompt)

    def dspy_metric(example, prediction):
        """Adapter: dspy.Evaluate expects a numeric score, not (score, feedback)."""
        return math_metric(example, prediction)[0]

    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=dspy_metric,
        num_threads=int(os.environ.get("AIME_EVAL_NUM_THREADS", 16)),
        display_progress=True,
        # Reasoning models can run out of tokens and return an unparseable
        # answer; count that as a wrong answer (score 0) instead of aborting.
        failure_score=0.0,
        max_errors=len(dataset),
    )

    eval_result = evaluator(predictor)
    return eval_result.score / 100.0
