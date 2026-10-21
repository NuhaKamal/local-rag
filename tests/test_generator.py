from rag_from_scratch.generator import (
    NO_ANSWER,
    Generator,
    build_context,
    build_messages,
    has_support,
)
from rag_from_scratch.models import Chunk

CHUNK = Chunk("Score of 3 means maintenance.", "book.pdf", 45, 0, "book.pdf:p45:c0")
HIT = (CHUNK, 0.9)  # (chunk, cosine score), the shape the retriever returns


def test_context_labels_each_chunk_with_its_page():
    context = build_context([HIT])
    assert "[1] (page 45)" in context
    assert "Score of 3 means maintenance." in context


def test_messages_have_a_system_and_a_user_part():
    messages = build_messages("What is a 3?", [HIT])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "What is a 3?" in messages[1]["content"]
    assert NO_ANSWER in messages[0]["content"]


def test_guard_compares_the_best_score_to_min_score():
    assert has_support([(CHUNK, 0.9)]) is True
    assert has_support([(CHUNK, -1.0)]) is False
    assert has_support([]) is False


def test_low_score_skips_the_model():
    generator = Generator.__new__(Generator)  # skips __init__, so no model is loaded
    out = generator.answer("anything", [(CHUNK, -1.0)])
    assert out["answer"] == NO_ANSWER
    assert out["sources"] == []


def test_no_hits_skips_the_model():
    generator = Generator.__new__(Generator)
    assert generator.answer("anything", [])["answer"] == NO_ANSWER