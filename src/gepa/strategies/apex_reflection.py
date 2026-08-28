# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Two-step Critique -> Mutate reflection (APEX Appendix C).

Faithful implementation of Algorithm 1 lines 6-7 of *APEX: Automated Prompt
Engineering eXpert with Dynamic Data Selection* (Wang et al.,
arXiv:2606.11459v1)::

    C     <- LLM_meta(Critique(P_curr, E))
    P_new <- LLM_meta(Mutate(P_curr, C))

Two separate meta-LLM calls, not one. The critique step diagnoses a root cause
and emits a structured directive (Locator / Diagnosis / Instruction); the mutate
step applies only that directive. GEPA's default reflection is a single call, so
this strategy replaces it.

Prompt templates are transcribed from Appendix C: Listing 4 (critique), Listing 5
(mutation), and Listing 6 (error case). The paper's placeholders are ``{prompt}``,
``{error_cases}``, and ``{feedback}``.

Batching
--------
``reflect_many`` issues exactly two batched rounds regardless of how many tasks
an iteration produced: all critiques go out together, then all mutations. This
preserves the strict critique-before-mutate ordering of lines 6-7 while keeping
the wall-clock cost of ``n`` parallel proposals at two round-trips.

Error-case rendering
--------------------
Listing 6 formats each failure as score / input / actual output / feedback. GEPA's
reflective dataset is a list of records per component whose keys are set by the
adapter; the built-in adapters use ``Inputs`` / ``Generated Outputs`` /
``Feedback``. Those are mapped onto Listing 6 when present, and any record that
does not expose them falls back to rendering its raw key-value pairs, so the
strategy works with adapters that name fields differently.

Deviations from the paper
-------------------------
* The paper optimizes a single prompt. GEPA candidates may hold several named
  components, so the two-step exchange runs once per component that has feedback,
  and all of those calls share the two batched rounds.
* When the critique output cannot be parsed, that component is skipped rather
  than mutated from an unstructured critique -- the mutate template requires a
  Locator/Diagnosis/Instruction triple to act on.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionJob, ReflectionProposal

# --- Appendix C, Listing 4: APEX Critique Prompt ---------------------------
CRITIQUE_PROMPT_TEMPLATE = """You are an expert prompt engineer. Your sole function is to analyze a faulty prompt and recommend the single most impactful, generalizable change for the next optimization iteration. Adhere strictly to the following process.

### The Prompt Under Analysis
The prompt being evaluated.
<current_prompt>
{prompt}
</current_prompt>

---

### Failure Case Analysis
Specific examples where the prompt produced a suboptimal output.
<failure_cases>
{error_cases}
</failure_cases>

---

### Instructions

**Step 1: Diagnose the Root Cause**
Analyze the **Failure Case Analysis** to identify the primary, underlying reasons for most of the errors. Classify the root cause into one of two major types:

* **Type 1: Weak Decision Boundaries (Defining the "What")**
    * **Ambiguity & Vague Definitions:** Terms, tone, or success criteria are open to interpretation.
    * **Constraint Loopholes:** Missing exclusionary constraints allow unwanted behaviors.
    * **Input Confusion:** The prompt lacks delimiters (e.g., XML tags, quotes), causing the model to confuse user input with instructions.
    * **Missing Context/Grounding:** The prompt fails to explicitly bind the model to provided source material (leading to hallucinations).
* **Type 2: Missing Process Instructions (Defining the "How")**
    * **Cognitive Overload:** The task is too complex for a single instruction and requires explicit decomposition (breaking the problem into distinct sub-tasks).
    * **Implicit Logic:** The prompt assumes the model knows the specific algorithm required to transform input to output.

**Step 2: Formulate the Recommendation**
Based on your diagnosis in Step 1, construct the recommendation according to these principles:

* **1. Apply the Correct Remediation Strategy:**
    * *If Type 1 (Boundary Issues):*
        * **Operationalize Definitions:** Change subjective adjectives to objective metrics.
        * **Enforce Delimiters:** Recommend wrapping input data in explicit tags (e.g., <input>...</input>) to separate it from instructions. Note that the placeholders and the corresponding input content are non-editable.
        * **Strengthen Constraints:** Add negative constraints or "Grounding" instructions (e.g., "Answer only using the provided text").
    * *If Type 2 (Process Issues):*
        * **Request Rationale:** Require the model to "show its work" or think step-by-step before answering. It is allowed to generate intermediate outputs before the final answer if a clear output format is specified for answer extraction.
        * **Decompose the Task:** Break the prompt into sequential, modular steps or sub-prompts.
        * **Few-Shot Prompting:** If the logic is abstract, strictly require "Input -> Output" examples to demonstrate the pattern.
* **2. Provide Generalizable Principles:**
    * Your recommendation must address the root cause, not just the specific failure examples.
    * **Crucially, do not quote or directly reference the provided `<failure_cases>`.** Your feedback must be independent of the specific content of the examples.
    * If an example is absolutely essential to illustrate your point, **you must invent a new, concise, and clear one** that demonstrates the principle effectively.

**Step 3: Construct the Feedback Object**
Translate your diagnosis into a structured directive for the Editor. You must use these exact fields:

* **Locator:** Quote the exact text, section header, or placeholder in the `<current_prompt>` where the fix should be applied.
* **Diagnosis:** Explain the specific weakness identified in Step 1 (e.g., "Type 1 Ambiguity: The adjective 'short' is subjective.").
* **Instruction:** The specific action for the Editor to take (e.g., "Replace 'short' with 'maximum 50 words'.").

---

### Required Output Format
You may provide explanatory text or rationale first (e.g., "Analysis: ...").
Your final response must be enclosed in `<actionable_feedback>` tags.
Inside these tags, you must strictly follow this format:

<actionable_feedback>
**Locator:** [Quote text or Header in the `<current_prompt>`]
**Diagnosis:** [Brief Type 1/Type 2 explanation]
**Instruction:** [Precise editing instruction]
</actionable_feedback>"""

# --- Appendix C, Listing 5: APEX Mutation Prompt ---------------------------
MUTATION_PROMPT_TEMPLATE = """You are an Adaptive Prompt Editor. Your goal is to rewrite a prompt based on targeted structured feedback.

### Input Data
**1. Original Prompt:**
<current_prompt>
{prompt}
</current_prompt>

**2. Critical Feedback:**
<feedback>
{feedback}
</feedback>
*(This feedback contains a Locator, a Diagnosis, and an Instruction.)*

---

### Execution Protocol

**1. Analysis Phase**
Before rewriting, explicitly plan your edit based on the input structure:
* **Target:** Find the specific text cited in the feedback's **Locator**.
* **Context:** Read the **Diagnosis** to understand the intent (this ensures you don't fix the grammar but miss the point).
* **Constraint Verification:** Ensure your planned edit does not accidentally remove critical constraints, negative instructions, or variable placeholders (e.g., `{{variable}}`).

**2. Revision Phase**
Rewrite the prompt using the following **"Logic vs. Syntax"** rules:
* **Variable Lockdown (Strict):** Treat all placeholders (e.g., `{{variable}}`) as immutable constants. Do not introduce any new placeholders, and do not add, remove, rename, or reformat existing ones. The content represented by the placeholder is non-editable.
* **Logic Lock (Strict):** Do not remove or alter instructions, constraints, or steps that are *not* targeted by the **Locator**.
* **Contextual Integration (Flexible):** You *are* permitted to adjust the wording, transitions, and grammar of the sentences surrounding the **Locator** to ensure the new changes blend naturally. The final result should read as a unified document.

---

### Output Format
**Part 1: Edit Strategy**
Provide a comprehensive, step-by-step plan. You must include:
* **The specific text/section you will modify.**
* **How you will rephrase it to satisfy the Instruction.**
* **How you will ensure surrounding transitions remain smooth.**
* **Confirmation that specific variables/constraints are preserved.**

**Part 2: Revised Prompt**
Output the full text of the revised prompt strictly enclosed within `<new_instruction>` tags."""

# --- Appendix C, Listing 6: APEX Error Case Template -----------------------
ERROR_CASE_TEMPLATE = """### Failure Example (Score: {score})
**1. Input Context:**
<input>
{query}
</input>
**2. Actual Model Output:**
<actual_output>
{response}
</actual_output>
**3. Evaluation Feedback (Why this failed):**
<critique>
{feedback}
</critique>"""

# Keys the built-in adapters use in the reflective dataset; mapped onto
# Listing 6's query / response / feedback slots when present.
_INPUT_KEYS = ("Inputs", "inputs", "input", "query")
_OUTPUT_KEYS = ("Generated Outputs", "generated_outputs", "output", "response")
_FEEDBACK_KEYS = ("Feedback", "feedback", "execution_feedback", "critique")
_SCORE_KEYS = ("Score", "score", "Scores (Higher is Better)", "scores")

# Per-field cap when an unrecognized field is appended to the feedback slot.
# Trajectories run to 10-25 KB each; at ``m`` cases per critique prompt, passing
# them whole would dominate the prompt and bury the diagnosis Listing 4 asks for.
_EXTRA_FIELD_CHAR_LIMIT = 2000

_FEEDBACK_RE = re.compile(r"<actionable_feedback>(.*?)</actionable_feedback>", re.DOTALL | re.IGNORECASE)
_INSTRUCTION_RE = re.compile(r"<new_instruction>(.*?)</new_instruction>", re.DOTALL | re.IGNORECASE)


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _truncate(value: Any, limit: int = _EXTRA_FIELD_CHAR_LIMIT) -> str:
    """Cap one field's rendered length, noting how much was dropped."""
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text)} chars total]"


def render_error_case(record: Mapping[str, Any]) -> str:
    """Render one reflective-dataset record as Listing 6.

    Listing 6 has three slots (query / response / feedback) plus a score, but an
    adapter is free to name its fields differently, and the fields it uses
    instead are often the only ones that tell two failures apart. Anything no
    slot consumed is therefore appended under its own key instead of dropped.

    This is not cosmetic. While ``execution_feedback`` was absent from
    ``_FEEDBACK_KEYS``, the feedback slot rendered as "N/A" for every AlfWorld
    case, and the two surviving slots carry no signal there: ``input`` is
    boilerplate shared by all tasks and ``output`` is empty whenever an episode
    times out. Distinct minibatches then produced byte-identical critique
    prompts, so the mutation step received identical feedback and returned
    identical candidates -- an entire iteration of parallel proposals
    collapsing onto one prompt.
    """
    query = _first_present(record, _INPUT_KEYS)
    response = _first_present(record, _OUTPUT_KEYS)
    feedback = _first_present(record, _FEEDBACK_KEYS)
    score = _first_present(record, _SCORE_KEYS)

    if query is None and response is None and feedback is None:
        body = "\n".join(f"{key}: {_truncate(value)}" for key, value in record.items())
        return f"### Failure Example\n{body}"

    # One key per slot is consumed -- the first present, matching
    # ``_first_present`` -- and the rest become diagnostics.
    consumed: set[str] = set()
    for keys in (_INPUT_KEYS, _OUTPUT_KEYS, _FEEDBACK_KEYS, _SCORE_KEYS):
        for key in keys:
            if key in record:
                consumed.add(key)
                break
    extras = [(key, value) for key, value in record.items() if key not in consumed]

    feedback_text = "N/A" if feedback is None else str(feedback)
    if extras:
        rendered = "\n".join(f"- {key}: {_truncate(value)}" for key, value in extras)
        feedback_text = f"{feedback_text}\n\nAdditional diagnostics:\n{rendered}"

    return ERROR_CASE_TEMPLATE.format(
        score="N/A" if score is None else score,
        query="N/A" if query is None else query,
        response="N/A" if response is None else response,
        feedback=feedback_text,
    )


def render_error_cases(records: Sequence[Mapping[str, Any]]) -> str:
    """Render every failure case for one component, in dataset order."""
    return "\n\n".join(render_error_case(record) for record in records)


class ApexTwoStepReflection:
    """Critique then Mutate, as two batched meta-LLM rounds (Algorithm 1 lines 6-7).

    Args:
        lm: The meta-LLM. When it exposes ``batch_complete`` (as
            :class:`gepa.lm.LM` does via ``litellm.batch_completion``) each round
            is issued as one concurrent request; otherwise calls run
            sequentially.
        critique_prompt_template: Override for Listing 4. Must contain
            ``{prompt}`` and ``{error_cases}``.
        mutation_prompt_template: Override for Listing 5. Must contain
            ``{prompt}`` and ``{feedback}``.
        logger: Optional logger for skipped components.

    Stateless: ``reflect`` returns ``self`` as the next reflection LM.
    """

    def __init__(
        self,
        lm: LanguageModel,
        critique_prompt_template: str | None = None,
        mutation_prompt_template: str | None = None,
        logger: Any | None = None,
    ):
        self.lm = lm
        self.critique_prompt_template = critique_prompt_template or CRITIQUE_PROMPT_TEMPLATE
        self.mutation_prompt_template = mutation_prompt_template or MUTATION_PROMPT_TEMPLATE
        self.logger = logger

        for name, template, placeholders in (
            ("critique_prompt_template", self.critique_prompt_template, ("{prompt}", "{error_cases}")),
            ("mutation_prompt_template", self.mutation_prompt_template, ("{prompt}", "{feedback}")),
        ):
            missing = [p for p in placeholders if p not in template]
            if missing:
                raise ValueError(f"Missing placeholder(s) in {name}: {', '.join(missing)}")

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)

    def _complete(self, prompts: list[str]) -> list[str]:
        """Issue one batched round of completions."""
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.lm(prompts[0])]
        batch_complete = getattr(self.lm, "batch_complete", None)
        if batch_complete is not None:
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
            return list(batch_complete(messages))
        return [self.lm(prompt) for prompt in prompts]

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, ApexTwoStepReflection]:
        """The N=1 case of :meth:`reflect_many`."""
        return self.reflect_many([(candidate, reflective_dataset, components_to_update)])[0]

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, ApexTwoStepReflection]]:
        """Run critique for every (job, component), then mutation for every survivor.

        Exactly two batched rounds, so ``n`` parallel proposals cost two
        round-trips rather than ``2n``.
        """
        # Round 1: critique. One entry per (job, component) with feedback.
        targets: list[tuple[int, str]] = []
        critique_prompts: list[str] = []
        for job_idx, (candidate, reflective_dataset, components_to_update) in enumerate(jobs):
            for name in components_to_update:
                records = reflective_dataset.get(name)
                if not records:
                    self._log(f"Component '{name}' is not in reflective dataset. Skipping.")
                    continue
                targets.append((job_idx, name))
                critique_prompts.append(
                    self.critique_prompt_template.format(
                        prompt=candidate[name],
                        error_cases=render_error_cases(records),
                    )
                )

        critique_outputs = self._complete(critique_prompts)

        # Parse the structured directive; a component whose critique cannot be
        # parsed is dropped rather than mutated from an unstructured critique.
        mutate_targets: list[tuple[int, str]] = []
        mutation_prompts: list[str] = []
        feedbacks: list[str] = []
        for (job_idx, name), raw in zip(targets, critique_outputs, strict=True):
            match = _FEEDBACK_RE.search(raw or "")
            if match is None:
                self._log(f"Component '{name}': critique returned no <actionable_feedback>; skipping.")
                continue
            feedback = match.group(1).strip()
            mutate_targets.append((job_idx, name))
            feedbacks.append(feedback)
            mutation_prompts.append(
                self.mutation_prompt_template.format(
                    prompt=jobs[job_idx][0][name],
                    feedback=feedback,
                )
            )

        # Round 2: mutation.
        mutation_outputs = self._complete(mutation_prompts)

        proposals = [ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}) for _ in jobs]
        for (job_idx, name), feedback, raw in zip(mutate_targets, feedbacks, mutation_outputs, strict=True):
            match = _INSTRUCTION_RE.search(raw or "")
            if match is None:
                self._log(f"Component '{name}': mutation returned no <new_instruction>; skipping.")
                continue
            proposal = proposals[job_idx]
            proposal.new_texts[name] = match.group(1).strip()
            # Record the mutation prompt for callbacks; the critique round's
            # intermediates go to metadata, which is where multi-call reflection
            # strategies are expected to put per-call detail.
            proposal.prompts[name] = self.mutation_prompt_template.format(
                prompt=jobs[job_idx][0][name], feedback=feedback
            )
            proposal.raw_lm_outputs[name] = raw
            proposal.metadata[f"apex:critique_feedback:{name}"] = feedback

        # Attach the raw critique outputs so nothing is lost when parsing failed.
        for (job_idx, name), raw in zip(targets, critique_outputs, strict=True):
            proposals[job_idx].metadata[f"apex:critique_raw:{name}"] = raw

        return [(proposal, self) for proposal in proposals]


__all__ = [
    "CRITIQUE_PROMPT_TEMPLATE",
    "ERROR_CASE_TEMPLATE",
    "MUTATION_PROMPT_TEMPLATE",
    "ApexTwoStepReflection",
    "render_error_case",
    "render_error_cases",
]
