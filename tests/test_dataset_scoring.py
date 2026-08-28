"""Tests for the pluggable scorers and the dataset split/cache layer.

Scorer tests are pure. Dataset tests avoid the network by monkeypatching the
HuggingFace ``load_dataset`` used inside ``examples.aime_math.datasets`` and by
pointing the cache dir at a tmp path.
"""

import json

import pytest

from examples.aime_math import datasets as ds
from examples.aime_math.scoring import get_scorer, score_int, score_latex

# --- scorers --------------------------------------------------------------


def test_score_int_exact_match():
    assert score_int("116", "116")[0] == 1.0
    assert score_int(116, 116)[0] == 1.0


def test_score_int_wrong_and_unparseable():
    assert score_int("116", "117")[0] == 0.0
    # A non-integer prediction (e.g. truncated reasoning) scores 0 with feedback.
    score, feedback = score_int("116", "not-a-number")
    assert score == 0.0
    assert "integer" in feedback.lower()


def test_score_latex_symbolic_equivalence():
    # 1/2 and 0.5 are symbolically equal via math_verify.
    assert score_latex("\\frac{1}{2}", "0.5")[0] == 1.0
    assert score_latex("116", "116")[0] == 1.0


def test_score_latex_normalized_fallback():
    # \left/\right and spacing differences match under normalization.
    assert score_latex("\\left(3\\right)", "(3)")[0] == 1.0


def test_score_latex_wrong():
    assert score_latex("\\frac{1}{2}", "\\frac{1}{3}")[0] == 0.0


def test_get_scorer_dispatch_and_unknown():
    assert get_scorer("int") is score_int
    assert get_scorer("latex") is score_latex
    with pytest.raises(ValueError):
        get_scorer("nope")


# --- dataset cache + splits ----------------------------------------------


def _fake_math500(n=10):
    return [
        {
            "problem": f"q{i}",
            "answer": str(i),
            "solution": f"sol{i}",
            "subject": "Algebra",
            "level": 1,
            "unique_id": f"id{i}",
        }
        for i in range(n)
    ]


def test_load_or_download_writes_and_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))
    calls = {"n": 0}

    def fake_load_dataset(hf_id, hf_config=None, split=None):
        calls["n"] += 1
        return _fake_math500()

    monkeypatch.setattr(ds, "load_dataset", fake_load_dataset)

    raw = ds.load_or_download("math500")
    assert calls["n"] == 1
    assert (tmp_path / "math500.json").exists()
    # Post-processing mapped raw fields onto AIME keys + preserved originals.
    rec = raw["all"][0]
    assert rec["input"] == "q0" and rec["answer"] == "0"
    assert rec["_subject"] == "Algebra" and rec["_unique_id"] == "id0"

    # Second call reads cache — no extra download.
    raw2 = ds.load_or_download("math500")
    assert calls["n"] == 1
    assert raw2["all"][0]["input"] == "q0"

    # Cached JSON is valid and round-trips.
    with open(tmp_path / "math500.json", encoding="utf-8") as f:
        assert json.load(f)["all"][0]["input"] == "q0"


def test_load_splits_trims_each_split(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **k: _fake_math500(10))

    # Per-split caps: train<=2, val<=1, test<=3.
    train, val, test = ds.load_splits("math500", seed=0, sizes=(2, 1, 3))
    assert len(train) <= 2 and len(val) <= 1 and len(test) <= 3
    # Splits are disjoint (no record appears twice).
    seen = [r["input"] for r in train + val + test]
    assert len(seen) == len(set(seen))


def test_load_splits_seed_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **k: _fake_math500(10))

    a = ds.load_splits("math500", seed=7)
    b = ds.load_splits("math500", seed=7)
    assert [r["input"] for r in a[0]] == [r["input"] for r in b[0]]


# --- HMMT (multi-source, latex) ------------------------------------------


def _fake_hmmt(hf_id, hf_config=None, split=None):
    """Stub HF loader dispatching on dataset id, mirroring each source's schema."""
    if hf_id == "FlagEval/HMMT_2025":  # trainset: question + solution
        return [{"id": i, "question": f"train_q{i}", "answer": i, "solution": f"sol{i}"} for i in range(30)]
    if hf_id == "MathArena/hmmt_feb_2026":  # val
        return [{"problem_idx": i, "problem": f"val_q{i}", "answer": f"\\frac{{1}}{{{i + 1}}}"} for i in range(33)]
    if hf_id == "MathArena/hmmt_feb_2023":
        return [{"problem_idx": i, "problem": f"t23_q{i}", "answer": i} for i in range(30)]
    if hf_id == "MathArena/hmmt_feb_2024":
        return [{"problem_idx": i, "problem": f"t24_q{i}", "answer": i} for i in range(30)]
    raise AssertionError(f"unexpected hf_id {hf_id!r}")


def test_hmmt_splits_sources_and_solution(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ds, "load_dataset", _fake_hmmt)

    train, val, test = ds.load_splits("hmmt", seed=0)

    # Sizes: train=30 (FlagEval), val=33 (2026), test=30+30 (2023+2024).
    assert len(train) == 30
    assert len(val) == 33
    assert len(test) == 60

    # Train comes from FlagEval: question mapped to input, solution preserved.
    assert all(r["input"].startswith("train_q") for r in train)
    assert all(r["solution"].startswith("sol") for r in train)

    # Val is hmmt_feb_2026; test is 2023 followed by 2024, and has no solution.
    assert all(r["input"].startswith("val_q") for r in val)
    assert any(r["input"].startswith("t23_q") for r in test)
    assert any(r["input"].startswith("t24_q") for r in test)
    assert all("solution" not in r for r in test)

    # Registered as latex grading.
    assert ds.get_spec("hmmt").answer_type == "latex"


def test_hmmt_answers_stringified(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ds, "load_dataset", _fake_hmmt)

    train, _, _ = ds.load_splits("hmmt", seed=0)
    # Integer HF answers are coerced to str so the latex scorer handles them.
    assert all(isinstance(r["answer"], str) for r in train)
