"""Dataset registry with local JSON caching.

Each dataset is described by a :class:`DatasetSpec` that knows (a) which
HuggingFace source(s) to pull, (b) how to map a raw item onto the AIME key
convention (``input`` / ``answer`` / optional ``solution``, plus any original
fields preserved under ``_``-prefixed keys), and (c) how to carve the loaded
records into train / val / test.

The raw, post-processed records are cached to ``data/<name>.json`` on first use;
subsequent runs read the local file instead of hitting the network. To add a
dataset, register a spec in :data:`SPECS` — nothing else in the pipeline changes.
"""

import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class Source:
    """One HuggingFace split feeding a dataset, plus its record mapper."""

    hf_id: str
    hf_split: str
    to_example: Callable[[dict], dict]
    hf_config: str | None = None


@dataclass
class DatasetSpec:
    """A named dataset: its sources, grading rule, and split policy."""

    name: str
    answer_type: str
    sources: dict[str, Source]
    # splitter(raw, seed, k) -> (train, val, test); raw maps source-key -> records.
    splitter: Callable[[dict[str, list[dict]], int, int | None], tuple[list, list, list]]
    # What the model receives as ``input``. "text" is the value verbatim;
    # "image" means the record holds a *relative path* that ``_to_example`` wraps
    # in a ``dspy.Image`` (records must stay JSON-serializable for the cache, so
    # the Image cannot be built here).
    input_type: str = "text"
    # Locally prepared dataset with no HuggingFace source: the cache file IS the
    # dataset. Makes a missing cache a clear error instead of a bare KeyError
    # from the splitter reading an empty ``raw``.
    local: bool = False


def _trim(records: list[dict], k: int | None) -> list[dict]:
    return records[:k] if k else records


# Per-split size caps: (train_k, val_k, test_k); each None/0 means "no cap".
Sizes = tuple[int | None, int | None, int | None]


def _apply_sizes(train, val, test, sizes: Sizes | None):
    if sizes is None:
        return train, val, test
    tk, vk, sk = sizes
    return _trim(train, tk), _trim(val, vk), _trim(test, sk)


def _split_two_source(raw, seed, sizes: Sizes | None):
    """AIME's original behaviour: train/val from one source (shuffled with a
    fixed seed 0 for reproducibility with the pre-refactor code), test from
    another. ``seed`` is intentionally ignored here to preserve the exact old
    ordering; MATH-500 and other single-source datasets honour it."""
    train_val = list(raw["train"])
    random.Random(0).shuffle(train_val)
    half = len(train_val) // 2
    train, val = train_val[:half], train_val[half:]
    test = list(raw["test"])
    return _apply_sizes(train, val, test, sizes)


def _split_hmmt(raw, seed, sizes: Sizes | None):
    """HMMT: train from the solution-carrying FlagEval/HMMT_2025 (shuffled by
    ``seed``), val from hmmt_feb_2026, test from hmmt_feb_2023 + hmmt_feb_2024
    concatenated. The three eval sources are distinct competition years from the
    training set, so evaluation stays disjoint from training."""
    train = list(raw["train"])
    random.Random(seed).shuffle(train)
    val = list(raw["val"])
    test = list(raw["test_2023"]) + list(raw["test_2024"])
    return _apply_sizes(train, val, test, sizes)


def _split_presplit(raw, seed, sizes: Sizes | None):
    """Records were already assigned to splits by a preparation script.

    Used by locally prepared datasets whose split policy is not expressible as a
    shuffle (e.g. pptblank groups by source .pptx to prevent template leakage,
    with a hand-picked deck assignment). ``seed`` is unused — the split is fixed
    — but ``sizes`` still applies so smoke tests can trim.
    """
    return _apply_sizes(list(raw["train"]), list(raw["val"]), list(raw["test"]), sizes)


def _make_single_source_splitter(ratios=(0.4, 0.3, 0.3)):
    """Split one source into train/val/test by ``ratios`` after a seeded shuffle."""

    def _split(raw, seed, sizes: Sizes | None):
        records = list(raw["all"])
        random.Random(seed).shuffle(records)
        n = len(records)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train = records[:n_train]
        val = records[n_train : n_train + n_val]
        test = records[n_train + n_val :]
        return _apply_sizes(train, val, test, sizes)

    return _split


# --- Record mappers (raw HF item -> AIME-keyed dict, original fields kept) ---


def _aime_train_example(item: dict) -> dict:
    return {
        "input": item["problem"],
        "answer": item["answer"],
        "solution": item.get("solution", ""),
        "_id": item.get("id"),
        "_url": item.get("url"),
    }


def _aime_test_example(item: dict) -> dict:
    return {
        "input": item["problem"],
        "answer": item["answer"],
        "_problem_idx": item.get("problem_idx"),
    }


def _hmmt_train_example(item: dict) -> dict:
    # FlagEval/HMMT_2025: field is ``question`` (not ``problem``) and carries a
    # worked ``solution`` that flows into reflection feedback.
    return {
        "input": item["question"],
        "answer": str(item["answer"]),
        "solution": item.get("solution", ""),
        "_id": item.get("id"),
    }


def _hmmt_eval_example(item: dict) -> dict:
    # MathArena/hmmt_feb_*: ``problem`` + ``answer`` only, no solution.
    return {
        "input": item["problem"],
        "answer": str(item["answer"]),
        "_problem_idx": item.get("problem_idx"),
        "_problem_type": item.get("problem_type"),
    }


def _math500_example(item: dict) -> dict:
    # Preserve every original field; expose the AIME keys the pipeline reads.
    return {
        "input": item["problem"],
        "answer": item["answer"],
        "solution": item.get("solution", ""),
        "_subject": item.get("subject"),
        "_level": item.get("level"),
        "_unique_id": item.get("unique_id"),
    }


SPECS: dict[str, DatasetSpec] = {
    "aime": DatasetSpec(
        name="aime",
        answer_type="int",
        sources={
            "train": Source("AI-MO/aimo-validation-aime", "train", _aime_train_example, hf_config="default"),
            "test": Source("MathArena/aime_2026", "train", _aime_test_example, hf_config="default"),
        },
        splitter=_split_two_source,
    ),
    "math500": DatasetSpec(
        name="math500",
        answer_type="latex",
        sources={
            "all": Source("HuggingFaceH4/MATH-500", "test", _math500_example),
        },
        splitter=_make_single_source_splitter(),
    ),
    "hmmt": DatasetSpec(
        name="hmmt",
        answer_type="latex",
        sources={
            # Only FlagEval/HMMT_2025 carries worked solutions -> use it as train.
            "train": Source("FlagEval/HMMT_2025", "train", _hmmt_train_example),
            "val": Source("MathArena/hmmt_feb_2026", "train", _hmmt_eval_example),
            "test_2023": Source("MathArena/hmmt_feb_2023", "train", _hmmt_eval_example),
            "test_2024": Source("MathArena/hmmt_feb_2024", "train", _hmmt_eval_example),
        },
        splitter=_split_hmmt,
    ),
    "pptblank": DatasetSpec(
        name="pptblank",
        answer_type="yesno",
        # No HF sources: produced by examples/aime_math/prepare_pptblank.py.
        sources={},
        splitter=_split_presplit,
        input_type="image",
        local=True,
    ),
}


def get_spec(name: str) -> DatasetSpec:
    if name not in SPECS:
        raise ValueError(f"Unknown dataset {name!r}; registered: {sorted(SPECS)}.")
    return SPECS[name]


def load_or_download(name: str) -> dict[str, list[dict]]:
    """Return the post-processed records for ``name`` as ``{source_key: [...]}``.

    Reads ``data/<name>.json`` if present; otherwise downloads each source,
    maps it to the AIME key convention, caches the result, and returns it.
    """
    spec = get_spec(name)
    cache_path = os.path.join(DATA_DIR, f"{name}.json")

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    # A local dataset has no sources to fall back on: without the cache the
    # splitter would read an empty ``raw`` and fail with a bare KeyError.
    if spec.local:
        raise FileNotFoundError(
            f"Dataset {name!r} is prepared locally and its data file is missing: {cache_path}\n"
            f"Generate it with: python -m examples.aime_math.prepare_{name}"
        )

    raw: dict[str, list[dict]] = {}
    for key, src in spec.sources.items():
        ds = load_dataset(src.hf_id, src.hf_config, split=src.hf_split)
        raw[key] = [src.to_example(item) for item in ds]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    return raw


def load_splits(name: str, seed: int = 0, sizes: Sizes | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Load ``name`` (from cache or HF) and return (train, val, test) dict lists.

    ``sizes`` (when set) is a ``(train_k, val_k, test_k)`` tuple capping each
    split to its first k records; each entry may be None/0 to leave that split
    uncapped.
    """
    spec = get_spec(name)
    raw = load_or_download(name)
    return spec.splitter(raw, seed, sizes)
