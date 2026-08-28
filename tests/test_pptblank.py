"""Tests for the pptblank (PPT excessive-whitespace) dataset path.

Covers the three-state yes/no scorer, the pre-split local dataset spec, and the
image-aware example/side_info construction. All pure or tmp_path based — no
network and no real slide files.
"""

import json

import pytest

from examples.aime_math import datasets as ds
from examples.aime_math.scoring import extract_boxed, get_scorer, parse_verdict, score_yesno

# --- \box{} extraction ----------------------------------------------------


@pytest.mark.parametrize(
    "pred",
    [
        r"\box{yes}",
        r"\box{ yes }",
        r"\box{Yes}",
        r"\boxed{yes}",          # the real LaTeX macro, emitted out of habit
        r"$\box{yes}$",          # math-mode wrapper
        r"\BOX{YES}",
        "The lower half is empty.\n\n\\box{yes}",
    ],
)
def test_parse_verdict_reads_boxed_positive(pred):
    assert parse_verdict(pred) == ("yes", True)


@pytest.mark.parametrize("pred", [r"\box{no}", r"\boxed{No}", "Spacing is even. \\box{no}"])
def test_parse_verdict_reads_boxed_negative(pred):
    assert parse_verdict(pred) == ("no", True)


def test_extract_boxed_takes_the_last_box():
    """Reasoning that restates the format before committing must be read as its
    final answer, not its first mention of the template."""
    text = r"I will answer \box{yes} or \box{no} depending on the layout. \box{no}"
    assert extract_boxed(text) == "no"
    assert parse_verdict(text) == ("no", True)


def test_boxed_non_verdict_is_a_parse_failure():
    """``\\box{maybe}`` is a genuine non-answer: do not scavenge the prose around it."""
    assert parse_verdict(r"It looks empty, so yes. \box{maybe}") == (None, True)
    assert score_yesno("yes", r"\box{maybe}")[0] == 0.0


def test_extract_boxed_returns_none_without_a_box():
    assert extract_boxed("yes") is None
    assert extract_boxed(None) is None


# --- yes/no scorer --------------------------------------------------------


@pytest.mark.parametrize("pred", [r"\box{yes}", r"\boxed{yes}", "yes", "Yes", "YES.", "  yes  ",
                                  "是", "有", "true", "1"])
def test_score_yesno_accepts_positive_forms(pred):
    assert score_yesno("yes", pred)[0] == 1.0


@pytest.mark.parametrize("pred", [r"\box{no}", r"\boxed{no}", "no", "No", "NO.", "否", "没有",
                                  "无", "false", "0"])
def test_score_yesno_accepts_negative_forms(pred):
    assert score_yesno("no", pred)[0] == 1.0


def test_unboxed_verdict_grades_but_is_flagged():
    """A formatting slip must not be scored as a wrong answer — but the drift has
    to reach the reflection LM so it can pin the output format down."""
    score, feedback = score_yesno("yes", "yes")
    assert score == 1.0
    assert "not wrapped in \\box{}" in feedback

    score, feedback = score_yesno("yes", r"\box{yes}")
    assert score == 1.0
    assert "\\box" not in feedback.replace("\\box{}", "")  # no complaint when correct


def test_boxed_wrong_verdict_still_reports_miss_or_false_alarm():
    _, fn = score_yesno("yes", r"\box{no}")
    _, fp = score_yesno("no", r"\box{yes}")
    assert "false negative" in fn.lower()
    assert "false positive" in fp.lower()
    # Correctly formatted, so no format complaint on top of the wrong verdict.
    assert "not wrapped" not in fn and "not wrapped" not in fp


def test_score_yesno_verdict_with_trailing_explanation():
    # The prompt asks for one word, but a verdict plus prose should still grade
    # rather than count as a parse failure.
    assert score_yesno("no", "no, the margins look normal")[0] == 1.0
    assert score_yesno("yes", "yes - large gap under the chart")[0] == 1.0


def test_score_yesno_wrong_verdict():
    assert score_yesno("yes", "no")[0] == 0.0
    assert score_yesno("no", "yes")[0] == 0.0


@pytest.mark.parametrize("pred", ["maybe", "", None, "banana", "3"])
def test_score_yesno_unparseable_scores_zero_with_format_complaint(pred):
    score, feedback = score_yesno("yes", pred)
    assert score == 0.0
    assert "\\box{yes}" in feedback and "\\box{no}" in feedback


@pytest.mark.parametrize(
    "pred",
    ["none", "not sure", "nothing stands out", "no idea", "nope",
     # Strict by design: a verdict needs punctuation after it, so this is a parse
     # failure rather than a guessed negative. Fails loudly, never silently right.
     "no excessive whitespace"],
)
def test_score_yesno_near_miss_words_are_not_a_negative_verdict(pred):
    """A bare prefix test would read all of these as a confident "no".

    That would silently convert a non-answer into a graded verdict — and against a
    negative gold it would even be scored *correct*, hiding a broken prompt.
    """
    assert score_yesno("no", pred)[0] == 0.0, f"{pred!r} was accepted as a 'no' verdict"


def test_score_yesno_either_is_always_correct():
    """Mild-only slides may reasonably be called a problem or not."""
    for pred in ("yes", "no", "banana", "", None):
        score, feedback = score_yesno("either", pred)
        assert score == 1.0
        assert "mild" in feedback.lower()


def test_score_yesno_distinguishes_miss_from_false_alarm():
    """The only channel through which F1's asymmetry reaches the reflection LM.

    GEPA averages per-example scores, so the score itself cannot encode "a miss
    costs more than a false alarm" — the feedback text has to.
    """
    _, fn = score_yesno("yes", "no")
    _, fp = score_yesno("no", "yes")
    assert "false negative" in fn.lower()
    assert "false positive" in fp.lower()
    assert fn != fp


def test_score_yesno_rejects_bad_gold():
    score, feedback = score_yesno("probably", "yes")
    assert score == 0.0
    assert "Dataset error" in feedback


def test_get_scorer_dispatches_yesno():
    assert get_scorer("yesno") is score_yesno


# --- pre-split local dataset spec ----------------------------------------


def _make_images(root, *relpaths):
    """Write real (tiny) JPEGs: dspy.Image reads the file to infer its type."""
    from PIL import Image as PILImage

    for rel in relpaths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.new("RGB", (8, 8), "white").save(path, "JPEG")


def _write_cache(tmp_path, train=2, val=2, test=2):
    payload = {
        "train": [
            {"id": f"t{i}", "input": f"images/t{i}.jpg", "answer": "yes" if i % 2 else "no",
             "_annotated_path": f"images_annotated/t{i}.jpg", "_style_en": "business_minimal",
             "_severity": ["severe"] if i % 2 else [], "_area_ratios": [0.05] if i % 2 else []}
            for i in range(train)
        ],
        "val": [
            {"id": f"v{i}", "input": f"images/v{i}.jpg", "answer": "no",
             "_annotated_path": f"images_annotated/v{i}.jpg", "_style_en": "tech_futuristic",
             "_severity": [], "_area_ratios": []}
            for i in range(val)
        ],
        "test": [
            {"id": f"s{i}", "input": f"images/s{i}.jpg", "answer": "either",
             "_annotated_path": f"images_annotated/s{i}.jpg", "_style_en": "flat_illustration",
             "_severity": ["mild"], "_area_ratios": [0.06]}
            for i in range(test)
        ],
    }
    with open(tmp_path / "pptblank.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


def test_pptblank_spec_registered_as_local_image():
    spec = ds.get_spec("pptblank")
    assert spec.input_type == "image"
    assert spec.answer_type == "yesno"
    assert spec.local is True
    assert spec.sources == {}


def test_existing_specs_keep_text_defaults():
    """The new fields must not change any pre-existing dataset."""
    for name in ("aime", "math500", "hmmt"):
        spec = ds.get_spec(name)
        assert spec.input_type == "text"
        assert spec.local is False


def test_presplit_loads_from_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("load_dataset must not be called for a local dataset")

    monkeypatch.setattr(ds, "load_dataset", boom)
    _write_cache(tmp_path, train=3, val=2, test=4)

    train, val, test = ds.load_splits("pptblank", seed=42)
    assert (len(train), len(val), len(test)) == (3, 2, 4)
    assert train[0]["id"] == "t0"


def test_presplit_honors_size_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))
    _write_cache(tmp_path, train=5, val=5, test=5)
    train, val, test = ds.load_splits("pptblank", seed=0, sizes=(2, 1, 3))
    assert (len(train), len(val), len(test)) == (2, 1, 3)


def test_presplit_ignores_seed(tmp_path, monkeypatch):
    """The split is fixed by the preparation script, so seed must not reorder it."""
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))
    _write_cache(tmp_path, train=4)
    a = ds.load_splits("pptblank", seed=1)[0]
    b = ds.load_splits("pptblank", seed=99)[0]
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_missing_cache_raises_actionable_error(tmp_path, monkeypatch):
    """Without this guard the empty ``sources`` dict surfaces as a bare KeyError."""
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc:
        ds.load_or_download("pptblank")
    assert "prepare_pptblank" in str(exc.value)


# --- image examples + side_info ------------------------------------------


def test_to_example_wraps_image_and_keeps_metadata(tmp_path, monkeypatch):
    import dspy

    from examples.aime_math import utils

    monkeypatch.setenv("PPTBLANK_IMAGE_ROOT", str(tmp_path))
    _make_images(tmp_path, "images/x.jpg", "images_annotated/x.jpg")
    record = {
        "id": "01_business_minimal_003", "input": "images/x.jpg", "answer": "yes",
        "_annotated_path": "images_annotated/x.jpg", "_style_en": "business_minimal",
        "_severity": ["severe"], "_area_ratios": [0.071],
    }
    ex = utils._to_example(record, "yesno", "image")
    assert isinstance(ex.input, dspy.Image)
    assert ex.answer == "yes" and ex.answer_type == "yesno"
    assert ex.id == "01_business_minimal_003"          # GEPA/eval id plumbing
    assert ex._style_en == "business_minimal"          # available to side_info
    assert set(ex.inputs().keys()) == {"input"}

    # Text datasets keep the value verbatim.
    text_ex = utils._to_example({"input": "1+1?", "answer": "2"}, "int", "text")
    assert text_ex.input == "1+1?"


def test_build_side_info_image_task(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from examples.aime_math import utils
    from gepa import Image as GepaImage

    monkeypatch.setenv("PPTBLANK_IMAGE_ROOT", str(tmp_path))
    _make_images(tmp_path, "images/a.jpg", "images_annotated/a.jpg")
    ex = utils._to_example(
        {"id": "07_tech_futuristic_002", "input": "images/a.jpg", "answer": "yes",
         "_annotated_path": "images_annotated/a.jpg", "_style_en": "tech_futuristic",
         "_severity": ["severe", "mild"], "_area_ratios": [0.071, 0.12]},
        "yesno", "image",
    )
    _, feedback = score_yesno("yes", "no")
    si = utils.build_side_info(ex, SimpleNamespace(answer="no", reasoning="looks fine"), 0.0, feedback)

    # The annotated slide must be a gepa.Image: only that wrapper is converted to
    # an image content part by the reflection prompt renderer.
    assert isinstance(si["Slide"], GepaImage)
    assert si["Slide"].path.endswith("images_annotated/a.jpg")
    # Style reaches the reflection LM; topic deliberately does not.
    assert si["DesignStyle"] == "tech_futuristic"
    assert not any("topic" in k.lower() for k in si)
    assert "severe (7.1% of the canvas)" in si["GroundTruthRegions"]
    assert si["ReferenceVerdict"] == "yes" and si["ModelVerdict"] == "no"
    # Bookkeeping stays under a _ prefix so the adapter strips it from the prompt.
    assert si["_image_id"] == "07_tech_futuristic_002"


def test_build_side_info_states_the_negative_case(tmp_path, monkeypatch):
    """Negatives are shown the plain slide; without this text the reflection LM
    cannot tell a clean slide from one whose annotations are missing."""
    from types import SimpleNamespace

    from examples.aime_math import utils

    monkeypatch.setenv("PPTBLANK_IMAGE_ROOT", str(tmp_path))
    _make_images(tmp_path, "images/n.jpg", "images_annotated/n.jpg")
    ex = utils._to_example(
        {"id": "n1", "input": "images/n.jpg", "answer": "no",
         "_annotated_path": "images_annotated/n.jpg", "_style_en": "flat_illustration",
         "_severity": [], "_area_ratios": []},
        "yesno", "image",
    )
    si = utils.build_side_info(ex, SimpleNamespace(answer="no", reasoning=""), 1.0, "correct")
    assert "NO excessive whitespace" in si["GroundTruthRegions"]


def test_build_side_info_text_task_unchanged():
    """Math datasets must keep the original four-key payload."""
    from types import SimpleNamespace

    import dspy

    from examples.aime_math import utils

    ex = dspy.Example(input="1+1?", answer="2", answer_type="int").with_inputs("input")
    si = utils.build_side_info(ex, SimpleNamespace(answer="2", reasoning="r"), 1.0, "ok")
    assert set(si) == {"score", "input", "output", "reasoning", "execution_feedback"}
    assert si["input"] == "1+1?"
