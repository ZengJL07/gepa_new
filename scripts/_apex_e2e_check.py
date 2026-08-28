"""Offline end-to-end check for the APEX strategies (no network, no API key).

Runs the full Algorithm 1 loop against a deterministic fake task whose outcomes
genuinely flip across the lineage, so the Mixed tier stays populated and Section
4.3's real L8-L14 path is exercised rather than the degenerate full-eval
fallback.

Asserts the properties that past bugs violated:
  * parent/child overlap on D_eval is complete (E1 actually re-scores P_curr)
  * D_req holds only ids P_curr has no outcome for (the `s = None` semantics)
  * P_curr advances rather than locking for the whole run
  * the seed keeps its full evaluation of D (Algorithm 1 line 2)

Usage:  .venv/bin/python scripts/_apex_e2e_check.py
"""

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.apex_candidate_selector import ApexCurrentBestSelector
from gepa.strategies.apex_eval_policy import ApexRankSensitivePolicy
from gepa.strategies.apex_reflection import ApexTwoStepReflection
from gepa.strategies.apex_sampling import ApexDynamicSampling
from gepa.strategies.apex_stratification import RejectedHistoryTracker, build_history, stratify

N_D = 16
D = [{"id": i} for i in range(N_D)]


class VolatileAdapter:
    """A prompt is a set of integer "rules"; each rule solves a residue class.

    A later rule shadows earlier ones on overlapping residues, so adding a rule
    can break examples that used to pass. That volatility is the point: it keeps
    the Mixed tier populated, which is what Section 4.3 needs in order to do
    anything at all.
    """

    def __init__(self):
        self.propose_new_texts = None

    @staticmethod
    def _rules(prompt: str) -> list[int]:
        return [int(tok) for tok in prompt.split() if tok.isdigit()]

    def _solves(self, prompt: str, example_id: int) -> bool:
        rules = self._rules(prompt)
        if not rules:
            return False
        # Only the LAST matching rule counts -> later rules shadow earlier ones.
        for rule in reversed(rules):
            if example_id % 4 == rule % 4:
                return example_id % 3 == rule % 3
        return False

    def evaluate(self, batch, candidate, capture_traces=False):
        prompt = candidate["sys"]
        scores = [1.0 if self._solves(prompt, ex["id"]) else 0.0 for ex in batch]
        outputs = [{"ok": s == 1.0} for s in scores]
        traj = (
            [
                {"Inputs": str(ex["id"]), "Generated Outputs": str(o), "Feedback": "ok" if s else "wrong"}
                for ex, o, s in zip(batch, outputs, scores, strict=True)
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traj)

    def make_reflective_dataset(self, candidate, eval_batch, components):
        return {c: list(eval_batch.trajectories or []) for c in components}


class ScriptedLM:
    """Two-round reflection stub: appends one new rule per mutation."""

    def __init__(self):
        self.calls = 0

    def batch_complete(self, messages_list):
        out = []
        for messages in messages_list:
            prompt = messages[0]["content"]
            if "Adaptive Prompt Editor" in prompt:
                self.calls += 1
                base = prompt.split("<current_prompt>")[1].split("</current_prompt>")[0].strip()
                out.append(f"<new_instruction>{base} {self.calls}</new_instruction>")
            else:
                out.append(
                    "<actionable_feedback>\n**Locator:** rules\n"
                    "**Diagnosis:** Type 1: too few rules.\n"
                    "**Instruction:** Add one rule.\n</actionable_feedback>"
                )
        return out

    def __call__(self, prompt):
        return self.batch_complete([[{"role": "user", "content": prompt}]])[0]


def main() -> None:
    history = RejectedHistoryTracker(perfect_score=1.0)
    policy = ApexRankSensitivePolicy(n_eval=4, alpha_0=0.2, beta=0.03, lookback=5, perfect_score=1.0, history=history)
    sampler = ApexDynamicSampling(n=2, minibatch_size=3, lookback=5, perfect_score=1.0, history=history)
    lm = ScriptedLM()

    # Watch P_curr across the run: a pointer that never moves is the bug that
    # locked one earlier run for 11 consecutive iterations.
    p_curr_seen: list[int] = []
    original = policy.get_best_program

    def traced(state):
        idx = original(state)
        p_curr_seen.append(idx)
        return idx

    policy.get_best_program = traced  # type: ignore[method-assign]

    res = gepa.optimize(
        seed_candidate={"sys": "rules 0"},
        trainset=D,
        valset=None,  # APEX needs a single D
        adapter=VolatileAdapter(),
        max_metric_calls=200,
        reflection_lm=lm,
        reflection_strategy=ApexTwoStepReflection(lm),
        sampling_strategy=sampler,
        val_evaluation_policy=policy,
        candidate_selection_strategy=ApexCurrentBestSelector(policy),
        use_merge=False,
        perfect_score=1.0,
        seed=0,
        display_progress_bar=False,
    )

    subs = res.val_subscores
    parents = res.parents

    print(f"candidates: {len(res.candidates)}")
    print("\n=== child vs parent overlap on D_eval ===")
    partial = 0
    for i in range(1, len(subs)):
        p = parents[i][0]
        if p is None:
            continue
        ov = set(subs[i]) & set(subs[p])
        flag = ""
        if len(ov) < min(len(subs[i]), len(subs[p])):
            flag = "  <-- PARTIAL"
            partial += 1
        print(f"  {i} <- {p}: |child|={len(subs[i])} |parent|={len(subs[p])} overlap={len(ov)}{flag}")
    print(f"\npartial-overlap pairs: {partial}")
    assert partial == 0, "E1 must re-score P_curr on the whole D_eval"

    print("\n=== D_req holds only ids P_curr has no outcome for ===")
    hist = build_history(subs, [], 1.0)
    strat = stratify(list(range(N_D)), hist, policy.current_best_idx, 5)
    d_req = strat.bucket("M", None)
    leaked = [i for i in d_req if i in subs[policy.current_best_idx]]
    print(f"  D_req={d_req}  leaked(known to P_curr)={leaked}")
    assert not leaked, "D_req must hold volatile *unknowns*"

    print("\n=== P_curr trajectory ===")
    print(f"  P_curr={policy.current_best_idx}  GEPAResult.best_idx={res.best_idx}")
    print(f"  distinct P_curr values seen: {sorted(set(p_curr_seen))}")
    assert len(set(p_curr_seen)) > 1, "P_curr locked for the entire run"

    print(f"\nseed coverage: {len(subs[0])}/{N_D}")
    assert len(subs[0]) == N_D, "seed must keep its full evaluation (L2)"
    print(f"alpha={policy.alpha:.4f}  degenerate_fallbacks={policy.degenerate_fallbacks}")
    print("\nALL APEX E2E CHECKS PASSED")


if __name__ == "__main__":
    main()
