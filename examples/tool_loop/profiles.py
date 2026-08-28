"""Task profiles: one declarative record per task, so adding a task is additive.

A :class:`TaskProfile` captures everything that varies across tasks — the loop
kind, how to load splits, how to score an episode, the seed prompt, how to build a
stateful session (env tasks only), and per-task budget/concurrency defaults. The
GEPA wiring, sampling, protocol, and episode loops never branch on task name; they
consume a resolved profile. Future tasks (e.g. HotPotQA) register one entry here.

Two loop kinds:
  "answer" -> task_env.run_episode      (terminate on <final>, pure-function tools)
  "env"    -> task_env.run_env_episode  (terminate on env done, stateful session)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# --- Seed prompts (the sole GEPA-optimized component) --------------------------

ANSWER_INITIAL_PROMPT = (
    "You can solve the task by calling tools. "
    'To call a tool, output exactly: <call name="TOOL">{"arg": value}</call>. '
    "To submit the answer, output: <final>ANSWER</final>. "
)

ENV_INITIAL_PROMPT = (
    "You are an agent interacting with an environment over multiple turns. "
    'Each turn, emit exactly one action as: <call name="step">{"action": "..."}</call>. '
    "Read the observation you get back and choose your next action. "
    "Keep going until the environment reports the task is complete."
)


@dataclass(frozen=True)
class TaskProfile:
    """Declarative description of how one task is loaded, run, and scored."""

    name: str
    kind: str  # "answer" | "env"
    load_splits: Callable[..., tuple[list, list, list]]  # (train_n,val_n,test_n,seed)->splits
    scorer: Callable[[Any, Any], tuple[float, str]]  # (episode, example)->(score, feedback)
    seed_prompt: str
    make_session: Callable[[Any], Any] | None = None  # required iff kind == "env"
    defaults: dict[str, int] = field(default_factory=dict)  # max_turns/max_tokens/max_workers/N

    def __post_init__(self) -> None:
        if self.kind not in ("answer", "env"):
            raise ValueError(f"TaskProfile {self.name!r}: kind must be 'answer' or 'env', got {self.kind!r}.")
        if self.kind == "env" and self.make_session is None:
            raise ValueError(f"TaskProfile {self.name!r}: kind='env' requires make_session.")


# --- Loader / session / scorer adapters ---------------------------------------
# Imports are local to keep offline tasks from importing HTTP/env code eagerly.


def _guess_loader(train_n: int, val_n: int, test_n: int, seed: int):
    from examples.tool_loop.datasets import load_splits

    return load_splits(train_n=train_n, val_n=val_n, test_n=test_n, seed=seed)


def _guess_scorer(episode, example):
    from examples.tool_loop.scoring import score_episode

    return score_episode(episode, example)


def _agentgym_loader(env_name: str):
    def _load(train_n: int, val_n: int, test_n: int, seed: int):
        from examples.tool_loop.agentgym_datasets import load_agentgym_splits

        return load_agentgym_splits(env_name, train_n=train_n, val_n=val_n, test_n=test_n, seed=seed)

    return _load


def _env_scorer(episode, example):
    from examples.tool_loop.agentgym_scoring import score_env_episode

    return score_env_episode(episode, example)


def _make_session(example):
    """Build the stateful EnvSession for one AgentGym example."""
    if example.env_name == "textcraft":
        from examples.tool_loop.envs.textcraft import TextCraftSession

        return TextCraftSession(example.server_base, example.env_index)
    if example.env_name == "sciworld":
        from examples.tool_loop.envs.sciworld import SciWorldSession

        return SciWorldSession(example.server_base, example.env_index)
    if example.env_name == "webshop":
        from examples.tool_loop.envs.webshop import WebShopSession

        return WebShopSession(example.server_base, example.env_index)
    from examples.tool_loop.envs.alfworld import AlfWorldSession

    return AlfWorldSession(example.server_base, example.env_index)


# --- The registry --------------------------------------------------------------

PROFILES: dict[str, TaskProfile] = {
    "guess": TaskProfile(
        name="guess",
        kind="answer",
        load_splits=_guess_loader,
        scorer=_guess_scorer,
        seed_prompt=ANSWER_INITIAL_PROMPT,
        defaults=dict(max_turns=6, max_tokens=8000, max_workers=15, train_n=12, val_n=8, test_n=8),
    ),
    "textcraft": TaskProfile(
        name="textcraft",
        kind="env",
        load_splits=_agentgym_loader("textcraft"),
        scorer=_env_scorer,
        seed_prompt=ENV_INITIAL_PROMPT,
        make_session=_make_session,
        # TextCraft: small text obs, short trajectories, has /close -> concurrency
        # is cheap. 15 to match AlfWorld and the math scripts; the binding
        # constraint is the solver API, not the env.
        defaults=dict(max_turns=20, max_tokens=16000, max_workers=15, train_n=12, val_n=8, test_n=8),
    ),
    "alfworld": TaskProfile(
        name="alfworld",
        kind="env",
        load_splits=_agentgym_loader("alfworld"),
        scorer=_env_scorer,
        seed_prompt=ENV_INITIAL_PROMPT,
        make_session=_make_session,
        # AlfWorld concurrency: 15, matching the math scripts.
        # This used to be 3, justified as "no /close route, so instances
        # accumulate". That reasoning was wrong on two counts. (a) Instances are
        # cheap: SingleAlfredTWEnv uses batch_size=1, which selects SyncBatchEnv,
        # so no subprocess is forked — it is one in-process TextWorld state over
        # a ~64-80KB game file. A previous run created 111 instances with zero
        # errors. (b) Accumulation is driven by the TOTAL number of episodes (each
        # one POSTs /create), not by how many run at once, so lowering
        # concurrency never reduced it.
        # The real bottleneck is per-turn LLM latency (up to 20 turns/episode),
        # which concurrency does hide. The env server itself serializes /step
        # (sync handler in one uvicorn process), so env work does not speed up —
        # but the API waiting, which dominates, does.
        # Budget is deliberately TIGHT to make efficient planning the thing under
        # test: 20 actions and 12287 total tokens. The token cap counts the
        # model's own reasoning, so verbose deliberation is penalized as a real
        # cost. Solved episodes used a median of 11 turns, while failures ran to
        # ~28 exhaustively searching containers — 20 sits between the two.
        defaults=dict(max_turns=20, max_tokens=12287, max_workers=15, train_n=12, val_n=8, test_n=8),
    ),
    "sciworld": TaskProfile(
        name="sciworld",
        kind="env",
        load_splits=_agentgym_loader("sciworld"),
        # Reuses the shared env scorer unchanged: SciWorldSession normalizes the
        # cumulative 0-100 score to [0, 1], so `reward >= 1.0` means "fully
        # solved" — the same binary criterion as AlfWorld/TextCraft.
        scorer=_env_scorer,
        seed_prompt=ENV_INITIAL_PROMPT,
        make_session=_make_session,
        # PLACEHOLDER budget: max_turns/max_tokens still need to be measured the
        # way AlfWorld's were (run wide, then set between the solved-episode and
        # failed-episode medians). ScienceWorld tasks are multi-step recipes and
        # likely need more turns than AlfWorld, so these are a starting point, not
        # a calibrated value.
        defaults=dict(max_turns=30, max_tokens=16000, max_workers=15, train_n=12, val_n=8, test_n=8),
    ),
    "webshop": TaskProfile(
        name="webshop",
        kind="env",
        load_splits=_agentgym_loader("webshop"),
        # Same shared scorer, same binary criterion: WebShopSession already reports
        # the environment's fractional reward in [0, 1], so `reward >= 1.0` means a
        # purchase that satisfied every requested attribute, option and the price
        # limit. Deliberately binary despite the reward being dense — for prompt
        # optimization what carries the signal is the reference trajectory in the
        # feedback, not reward granularity.
        scorer=_env_scorer,
        seed_prompt=ENV_INITIAL_PROMPT,
        make_session=_make_session,
        # Expert trajectories are short: median 5 actions, p90 6, max 8
        # (search, click product, click each option, Buy Now). 15 leaves generous
        # room to browse and recover without inviting unbounded wandering.
        defaults=dict(max_turns=15, max_tokens=16000, max_workers=15, train_n=12, val_n=8, test_n=8),
    ),
    # Future — additive, no loop/sampling/main changes needed:
    # "hotpotqa": TaskProfile(
    #     name="hotpotqa", kind="answer",
    #     load_splits=load_hotpotqa_splits, scorer=score_qa_episode,
    #     seed_prompt=HOTPOTQA_PROMPT,
    #     defaults=dict(max_turns=8, max_tokens=12000, max_workers=10, train_n=..., ...),
    # ),
}


def get_profile(name: str) -> TaskProfile:
    """Look up a profile by task name (raises with the valid set on miss)."""
    key = (name or "").lower()
    if key not in PROFILES:
        raise ValueError(f"Unknown TOOL_LOOP_TASK={name!r}; expected one of {sorted(PROFILES)}.")
    return PROFILES[key]
