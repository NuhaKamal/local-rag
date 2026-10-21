"""
generator.py: Stage 6. Build the prompt (augment) and generate the answer.

CONTRACT:
  INPUT : question + hits, a list of (Chunk, score) from the retriever, best first
  OUTPUT: {"answer": str, "sources": list[dict], "prompt_tokens": int}

HOW IT WORKS:
  1. Guard: if the best score is below MIN_SCORE, answer "I don't know" without
     calling the model (retrieval always returns K chunks, even when none helps).
  2. Fit: drop the weakest chunks if the prompt would overflow the context window.
  3. Augment: paste numbered, page-labelled chunks and the question into a prompt
     with rules ("answer ONLY from the context").
  4. Generate: TinyLlama (through llama.cpp) writes the answer.

WHY THESE CHOICES:
  Grounding instruction : a small model will invent facts unless told not to.
  MIN_SCORE guard       : skips the model for unanswerable questions.
  Chat template         : TinyLlama-Chat expects Zephyr-style markers (chat_format).
  Low temperature       : faithful extraction, not creativity.

WHY build_messages IS SEPARATE FROM THE MODEL:
  build_messages only builds the messages; it does not run the model. Keeping it
  separate means you can print the exact prompt and see what the model sees,
  which is the best way to debug a bad answer.

RUN IT ON ITS OWN:
  uv run python -m rag_from_scratch.generator
  (shows the prompt and the guard without a model; generates if the model is in models/)
"""

import logging

from rag_from_scratch.config import (
    LLM_MODEL_PATH,
    MAX_NEW_TOKENS,
    MIN_SCORE,
    N_CTX,
    TEMPERATURE,
)
from rag_from_scratch.models import Chunk

logger = logging.getLogger(__name__)

Hit = tuple[Chunk, float]  # one retrieved chunk and its cosine score

NO_ANSWER = "I don't know based on the document."

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the question using ONLY the context provided. "
    f"If the context does not contain the answer, reply exactly: {NO_ANSWER} "
    "Be concise. Mention the page numbers you used, like (p. 12)."
)


def build_context(hits: list[Hit]) -> str:
    """Number each chunk and label it with its page, so the model can cite it."""
    blocks = []
    for i, (chunk, _score) in enumerate(hits, start=1):
        blocks.append(f"[{i}] (page {chunk.page_number})\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    """The 'augment' step: the exact messages the model will see."""
    user_text = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def has_support(hits: list[Hit]) -> bool:
    """The guard: is the best retrieved chunk similar enough to be worth answering from?"""
    return bool(hits) and hits[0][1] >= MIN_SCORE


def _refuse(reason: str) -> dict:
    logger.info("model skipped: %s", reason)
    return {"answer": NO_ANSWER, "sources": [], "prompt_tokens": 0}


class Generator:
    """Wraps TinyLlama. Call .answer(question, hits)."""

    def __init__(self, model_path=LLM_MODEL_PATH):
        from llama_cpp import Llama  # imported here: heavy, and only needed when the model loads

        logger.info("loading language model %s", model_path.name)
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=N_CTX,
            chat_format="zephyr",
            verbose=False,
        )

    def _fit_to_budget(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Drop the lowest-ranked chunks until prompt + answer fit in N_CTX."""
        budget = N_CTX - MAX_NEW_TOKENS - 64  # 64 = margin for template tokens that are added invisibly
        original = len(hits)
        while hits:
            messages = build_messages(question, hits)
            text = "\n".join(m["content"] for m in messages)
            if len(self.llm.tokenize(text.encode("utf-8"))) <= budget:
                break
            hits = hits[:-1]  # hits are sorted best-first, so drop the last
        if len(hits) < original:
            logger.warning("prompt too long: kept %d of %d chunks", len(hits), original)
        return hits

    def answer(self, question: str, hits: list[Hit]) -> dict:
        """Return {"answer", "sources", "prompt_tokens"}."""
        # Guard 1: nothing relevant retrieved -> don't ask the model at all
        if not has_support(hits):
            return _refuse(f"best score below MIN_SCORE ({MIN_SCORE})")

        # Guard 2: make sure the prompt fits the context window
        hits = self._fit_to_budget(question, hits)
        if not hits:
            return _refuse("no chunk fits the context window")

        logger.info("generating from %d chunks", len(hits))
        output = self.llm.create_chat_completion(
            messages=build_messages(question, hits),
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
        text = output["choices"][0]["message"]["content"].strip()
        sources = [
            {"source": c.source, "page_number": c.page_number, "score": s}
            for c, s in hits
        ]
        tokens = output["usage"]["prompt_tokens"]
        logger.info("generated %d characters (prompt was %d tokens)", len(text), tokens)
        return {"answer": text, "sources": sources, "prompt_tokens": tokens}


def _demo() -> None:
    question = "What does a score of 3 mean?"
    hits = [
        (Chunk("Score of 3: Maintenance. Keep practising the movement.",
               "book.pdf", 45, 0, "book.pdf:p45:c0"), 0.71),
        (Chunk("A functional movement is a pattern used in daily life.",
               "book.pdf", 12, 0, "book.pdf:p12:c0"), 0.52),
    ]

    print("== generator.py demo ==")
    print(f"INPUT : question = {question!r}")
    print(f"        {len(hits)} retrieved chunks (scores {[s for _, s in hits]})\n")

    print("STEP 1, the augmented prompt (no model needed):")
    for m in build_messages(question, hits):
        print(f"  [{m['role']}]")
        for line in m["content"].splitlines():
            print(f"    {line}")

    print(f"\nSTEP 2, the guard (MIN_SCORE = {MIN_SCORE}):")
    weak = [(hits[0][0], 0.10)]
    print(f"  best score 0.71 -> has_support = {has_support(hits)}")
    print(f"  best score 0.10 -> has_support = {has_support(weak)}  (the model would be skipped)")

    print("\nSTEP 3, generation:")
    if not LLM_MODEL_PATH.exists():
        print(f"  Model not found: {LLM_MODEL_PATH} (download it first, see the README)")
        return
    out = Generator().answer(question, hits)
    print(f"OUTPUT: answer        = {out['answer']!r}")
    print(f"        sources       = {out['sources']}")
    print(f"        prompt_tokens = {out['prompt_tokens']}")


if __name__ == "__main__":
    _demo()