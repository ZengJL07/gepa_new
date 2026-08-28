"""AgentGym (TextCraft / AlfWorld / ScienceWorld) examples + item_id -> env-index mapping.

AgentGym datasets do NOT carry env state; the env server materializes a task from
an integer index (TextCraft ``data_idx``, AlfWorld ``game``, ScienceWorld
``data_idx``). This module resolves each dataset ``item_id`` to that integer and
packages it as an :class:`EnvExample`.

Mapping rules (DATASETS.md §4):
  AlfWorld train ids ``"<task_type>_<task_id>"`` -> lookup in ``mappings_train.json``.
  AlfWorld test  ids ``"alfworld_2420"``        -> tail integer, used directly.
  TextCraft train ids ``"textcraft_31"``        -> lookup in ``textcraft_train_idx_remap.json``
                                                   (machine-specific! rebuild per host).
  TextCraft test  ids ``"textcraft_5"``          -> tail integer (drift risk; see §4).
  ScienceWorld ids ``"sciworld_536"``            -> tail integer, both splits.
  WebShop ids ``"webshop_6"``                     -> tail integer (goal index), both.

Official splits are train/test only; the validation split is carved from train
(DATASETS.md §6) with a seeded shuffle.
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_DATA_ROOT = "/home/jlzeng/code/AgentGym"
_DEFAULT_PORTS = {"textcraft": 36001, "alfworld": 36002, "sciworld": 36003, "webshop": 36004}


@dataclass
class EnvExample:
    """One AgentGym task instance, pre-resolved to a concrete env index."""

    env_name: str  # "textcraft" | "alfworld" | "sciworld" | "webshop"
    item_id: str
    env_index: int  # data_idx (TextCraft/ScienceWorld) or game (AlfWorld)
    server_base: str
    instruction: str = ""
    info: dict[str, Any] = field(default_factory=dict)
    # Reference solution for THIS task instance: the expert action sequence from
    # the dataset's own trajectory, observations stripped. Empty for test items,
    # which ship without trajectories.
    expert_actions: list[str] = field(default_factory=list)

    @property
    def input(self) -> str:
        """First user message is supplied by ``session.reset()``; instruction seeds it."""
        return self.instruction

    def with_inputs(self, *_names):
        return self


def _tail_int(item_id: str) -> int:
    return int(str(item_id).rsplit("_", 1)[-1])


def _default_data_root() -> str:
    return os.environ.get("TOOL_LOOP_DATA_ROOT", _DEFAULT_DATA_ROOT)


def _default_server_base(env_name: str) -> str:
    override = os.environ.get("TOOL_LOOP_ENV_SERVER")
    if override:
        return override.rstrip("/")
    port = _DEFAULT_PORTS[env_name]
    return f"http://127.0.0.1:{port}"


def _alfworld_train_lookup(mappings_path: str) -> dict[str, int]:
    """Build ``"<task_type>_<task_id>" -> item_id`` from mappings_train.json."""
    with open(mappings_path) as f:
        rows = json.load(f)
    return {f"{m['task_type']}_{m['task_id']}": int(m["item_id"]) for m in rows}


def _resolve_index(env_name: str, item_id: str, *, is_test: bool, data_root: str) -> int:
    """Map a dataset item_id to the integer env index the server expects."""
    if env_name == "alfworld":
        if is_test:
            return _tail_int(item_id)
        path = os.environ.get(
            "AGENTGYM_ALFWORLD_MAPPINGS",
            os.path.join(data_root, "agentenv-alfworld/configs/mappings_train.json"),
        )
        return _alfworld_train_lookup(path)[item_id]
    if env_name == "textcraft":
        if is_test:
            # No trajectories for test ids; tail number == local data_idx (drift risk, §4).
            return _tail_int(item_id)
        path = os.environ.get(
            "AGENTGYM_TEXTCRAFT_REMAP",
            os.path.join(data_root, "data/textcraft_train_idx_remap.json"),
        )
        with open(path) as f:
            remap = json.load(f)
        return int(remap[item_id])
    if env_name == "webshop":
        # The tail integer indexes the server's goals list directly
        # (web_agent_text_env.py: goal = self.goals[idx]). That list is built from
        # the FULL product dump; the 1000-product subset yields only ~13 goals and
        # cannot serve ids up to 6904.
        return _tail_int(item_id)
    if env_name == "sciworld":
        # Verified exhaustively against the server on this host: for all 2120
        # train items, games[tail_int] reproduces the task description embedded in
        # the item's own conversations. So no remap table is needed (unlike
        # TextCraft). Server-side games[] has 4639 entries; train ids reach 4632
        # and test ids 4336, both in range.
        return _tail_int(item_id)
    raise ValueError(f"Unknown env_name {env_name!r}; expected 'textcraft', 'alfworld', 'sciworld' or 'webshop'.")


def _load_items(env_name: str, split: str, data_root: str) -> list[str]:
    """Read the raw ``item_id`` list from an AgentGym split JSON."""
    sub = "train" if split == "train" else "test"
    path = os.path.join(data_root, "data", sub, f"{env_name}_{sub}.json")
    with open(path) as f:
        rows = json.load(f)
    return [r["item_id"] for r in rows]


def _load_expert_actions(env_name: str, split: str, data_root: str) -> dict[str, list[str]]:
    """``item_id -> expert action sequence`` from a split's ``conversations``.

    The official train files carry a reference solution per item as a
    human/gpt dialogue; the test files carry only ``item_id``. We keep just the
    gpt turns (the actions) and drop the human turns (the observations): the
    actions are what a prompt can be taught to produce, while replaying pages of
    observations would swamp the reflection prompt.

    Returns an empty mapping when the split has no ``conversations`` (test), so
    callers need no special case.
    """
    sub = "train" if split == "train" else "test"
    path = os.path.join(data_root, "data", sub, f"{env_name}_{sub}.json")
    with open(path) as f:
        rows = json.load(f)

    out: dict[str, list[str]] = {}
    for row in rows:
        actions = [
            _clean_expert_action(m.get("value", ""))
            for m in row.get("conversations", [])
            if m.get("from") == "gpt"
        ]
        # Every item in all three datasets opens with the same boilerplate
        # acknowledgement ("OK. I'll follow your instructions...") rather than an
        # action; verified 2120/2420/374 of 2120/2420/374 items.
        actions = [a for a in actions if a and not _is_acknowledgement(a)]
        if actions:
            out[row["item_id"]] = actions
    return out


def _is_acknowledgement(action: str) -> bool:
    return action.lower().startswith("ok. i'll follow your instructions")


def _clean_expert_action(raw: str) -> str:
    """Strip the trajectory's ``Action:``/``Thought: ...`` wrapper down to the action.

    Recorded gpt turns look like ``"Thought: ...\\nAction: go to kitchen"`` (or just
    the bare action). Only the action is portable into feedback.
    """
    text = str(raw or "").strip()
    marker = "Action:"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    return " ".join(text.split())


def _instruction_for(env_name: str) -> str:
    if env_name == "textcraft":
        from examples.tool_loop.envs.textcraft import _INSTRUCTION

        return _INSTRUCTION
    if env_name == "sciworld":
        from examples.tool_loop.envs.sciworld import _INSTRUCTION

        return _INSTRUCTION
    if env_name == "webshop":
        from examples.tool_loop.envs.webshop import _INSTRUCTION

        return _INSTRUCTION
    from examples.tool_loop.envs.alfworld import _INSTRUCTION

    return _INSTRUCTION


def load_agentgym_splits(
    env_name: str,
    *,
    train_n: int = 12,
    val_n: int = 8,
    test_n: int = 8,
    seed: int = 0,
    server_base: str | None = None,
    data_root: str | None = None,
) -> tuple[list[EnvExample], list[EnvExample], list[EnvExample]]:
    """Return (train, val, test) EnvExample lists.

    The official train split is shuffled once (seeded) and sliced into train/val
    (DATASETS.md §6); the official test split supplies the test examples.

    Sizes accept a "whole split" sentinel for formal evaluation:
    ``val_n <= 0`` drops the val holdout, ``train_n <= 0`` takes all remaining
    train, and ``test_n <= 0`` takes the entire official test split. Available
    upper bounds on this host: TextCraft train=374 / test=100, AlfWorld
    train=2420 / test=200, ScienceWorld train=2059 (2120 official minus the 61
    ids shared with test) / test=200.
    """
    env_name = env_name.lower()
    root = data_root or _default_data_root()
    base = (server_base or _default_server_base(env_name)).rstrip("/")
    instruction = _instruction_for(env_name)

    # Reference solutions live only in the train file. Loaded once (the sciworld
    # train JSON is ~25MB) rather than per example.
    expert_by_id = _load_expert_actions(env_name, "train", root)

    def _make(item_id: str, is_test: bool) -> EnvExample:
        return EnvExample(
            env_name=env_name,
            item_id=item_id,
            env_index=_resolve_index(env_name, item_id, is_test=is_test, data_root=root),
            server_base=base,
            instruction=instruction,
            expert_actions=list(expert_by_id.get(item_id, ())),
        )

    all_test_ids = _load_items(env_name, "test", root)

    # Drop train items that also appear in the official test split, BEFORE the
    # shuffle, so the exclusion does not depend on the seed.
    #
    # ScienceWorld's two official files share 61 item_ids. Since val is carved out
    # of train, those ids would be optimized on and then scored on: at
    # TRAIN_SIZE=40 / VAL_SIZE=45, 92 of 100 seeds leak at least one item (mean
    # 2.4, worst 6) — seed 42 leaks 3. That is test contamination, not a sampling
    # artifact, and drawing more train items only leaks more.
    # No-op for AlfWorld and TextCraft, whose id spaces do not intersect.
    train_ids = _load_items(env_name, "train", root)
    test_id_set = set(all_test_ids)
    train_ids = [i for i in train_ids if i not in test_id_set]

    random.Random(seed).shuffle(train_ids)

    # Size convention (so formal eval can request the whole split):
    #   val_n  <= 0 -> no validation holdout
    #   train_n<= 0 -> all remaining train after the val holdout
    #   test_n <= 0 -> the entire official test split
    val_count = max(0, val_n)
    val_ids = train_ids[:val_count]
    rest = train_ids[val_count:]
    remaining_train = rest if train_n <= 0 else rest[:train_n]

    test_ids = all_test_ids if test_n <= 0 else all_test_ids[:test_n]

    train = [_make(i, is_test=False) for i in remaining_train]
    val = [_make(i, is_test=False) for i in val_ids]
    test = [_make(i, is_test=True) for i in test_ids]
    return train, val, test
