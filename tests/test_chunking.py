"""Chunking tests — the 413-avoidance guarantee (Phase III)."""
from src.llm.chunking import count_tokens, salient_truncate


def test_truncate_respects_budget():
    text = "word " * 5000
    out = salient_truncate(text, max_tokens=200)
    assert count_tokens(out) <= 200


def test_short_text_untouched():
    text = "a short document"
    assert salient_truncate(text, max_tokens=1000) == text


def test_head_and_tail_preserved():
    text = "HEADSTART " + ("filler " * 2000) + "TAILEND"
    out = salient_truncate(text, max_tokens=100)
    assert "HEADSTART" in out and "TAILEND" in out
