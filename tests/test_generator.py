from local_rag.generation.generator import NO_ANSWER, Generator, build_context, build_messages

RESULT = {
    "text": "Score of 3 means maintenance.",
    "source": "book.pdf",
    "page_number": 45,
    "score": 0.9,
}


def test_context_labels_each_chunk_with_its_page():
    context = build_context([RESULT])
    assert "[1] (page 45)" in context
    assert "Score of 3 means maintenance." in context


def test_messages_have_a_system_and_a_user_part():
    messages = build_messages("What is a 3?", [RESULT])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "What is a 3?" in messages[1]["content"]
    assert NO_ANSWER in messages[0]["content"]  # prompt and code share one refusal sentence


def test_low_score_skips_the_model():
    generator = Generator.__new__(Generator)  # skips __init__, so no model is loaded
    out = generator.answer("anything", [{**RESULT, "score": -1.0}])  # lowest possible score
    assert out["answer"] == NO_ANSWER
    assert out["sources"] == []


def test_no_results_skips_the_model():
    generator = Generator.__new__(Generator)
    assert generator.answer("anything", [])["answer"] == NO_ANSWER