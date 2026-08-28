"""Both eval prompts are independently specifiable, by file or inline string.

The baseline used to accept only an inline env string while the optimized side
accepted a file, so comparing two GEPA artifacts meant pasting a whole prompt into
the environment. Both now resolve the same way: file > inline > default.
"""

import pytest

from examples.tool_loop.eval_dataset import _resolve_prompt


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("P_FILE", "P_TEXT"):
        monkeypatch.delenv(var, raising=False)


def test_default_when_nothing_is_set():
    assert _resolve_prompt("P_FILE", "P_TEXT", "SEED") == "SEED"


def test_inline_string_overrides_the_default(monkeypatch):
    monkeypatch.setenv("P_TEXT", "inline prompt")
    assert _resolve_prompt("P_FILE", "P_TEXT", "SEED") == "inline prompt"


def test_file_wins_over_inline(monkeypatch, tmp_path):
    """A file must win, so pointing at a GEPA artifact does not require first
    unsetting a leftover inline override."""
    path = tmp_path / "p.txt"
    path.write_text("from file\n")
    monkeypatch.setenv("P_TEXT", "inline prompt")
    monkeypatch.setenv("P_FILE", str(path))
    assert _resolve_prompt("P_FILE", "P_TEXT", "SEED") == "from file"


def test_trailing_newlines_are_stripped(monkeypatch, tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("prompt body\n\n")
    monkeypatch.setenv("P_FILE", str(path))
    assert _resolve_prompt("P_FILE", "P_TEXT", "SEED") == "prompt body"


@pytest.mark.parametrize("body", ["", "\n", "   \n\t "])
def test_empty_file_is_an_error_not_a_silent_fallback(monkeypatch, tmp_path, body):
    """Scoring the seed prompt while believing you scored an optimized one is
    worse than failing outright."""
    path = tmp_path / "p.txt"
    path.write_text(body)
    monkeypatch.setenv("P_FILE", str(path))
    with pytest.raises(ValueError, match="is empty"):
        _resolve_prompt("P_FILE", "P_TEXT", "SEED")


def test_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("P_FILE", str(tmp_path / "nope.txt"))
    with pytest.raises(FileNotFoundError):
        _resolve_prompt("P_FILE", "P_TEXT", "SEED")
