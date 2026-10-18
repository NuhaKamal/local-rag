"""
generation/generator.py — Stages 5 and 6: build the prompt (augment) and generate.

HOW IT WORKS:
  1. Take the top-K chunks returned by the retriever.
  2. Paste them into the prompt as numbered, page-labelled context.
  3. Instruct the model: answer ONLY from the context, otherwise say you don't know.
  4. Run TinyLlama through llama.cpp and return the answer plus its sources.

WHY THESE CHOICES:
  Grounding instruction : a small model will happily invent facts; the
                          system prompt tells it to stay inside the context.
  MIN_SCORE guard       : retrieval always returns K chunks, even for an
                          unanswerable question. If the best score is too low
                          we skip the model entirely rather than risk a
                          confident-sounding invention.
  Token budget          : prompt + answer must fit in N_CTX. If the chunks
                          are too long we drop the lowest-ranked ones.
  Chat template         : TinyLlama-Chat expects Zephyr-style markers; the
                          chat_format setting builds them for us.
  Low temperature       : we want faithful extraction, not creativity.
"""

from llama_cpp import Llama

from local_rag.config import (
    LLM_MODEL_PATH,
    MAX_NEW_TOKENS,
    MIN_SCORE,
    N_CTX,
    TEMPERATURE,
)

NO_ANSWER = "I don't know based on the document."

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the question using ONLY the context provided. "
    f"If the context does not contain the answer, reply exactly: {NO_ANSWER} "
    "Be concise. Mention the page numbers you used, like (p. 12)."
)


def build_context(results: list[dict]) -> str:
    """Number each chunk and label it with its page, so the model can cite it."""
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] (page {r['page_number']})\n{r['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[dict]) -> list[dict]:
    """The 'augment' step: the exact messages the model will see."""
    user_text = f"Context:\n{build_context(results)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


class Generator:
    """Wraps TinyLlama. Call .answer(question, retrieved_chunks)."""

    def __init__(self, model_path=LLM_MODEL_PATH):
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=N_CTX,
            chat_format="zephyr",
            verbose=False,
        )

    def _fit_to_budget(self, question: str, results: list[dict]) -> list[dict]:
        """Drop the lowest-ranked chunks until prompt + answer fit in N_CTX."""
        budget = N_CTX - MAX_NEW_TOKENS - 64  # 64 = margin for template tokens
        while results:
            messages = build_messages(question, results)
            text = "\n".join(m["content"] for m in messages)
            if len(self.llm.tokenize(text.encode("utf-8"))) <= budget:
                return results
            results = results[:-1]  # results are sorted best-first, so drop the last
        return results

    def answer(self, question: str, results: list[dict]) -> dict:
        """Return {"answer", "sources", "prompt_tokens"}."""
        # Guard 1: nothing relevant retrieved -> don't ask the model at all
        if not results or results[0]["score"] < MIN_SCORE:
            return {"answer": NO_ANSWER, "sources": [], "prompt_tokens": 0}

        # Guard 2: make sure the prompt fits the context window
        results = self._fit_to_budget(question, results)
        if not results:
            return {"answer": NO_ANSWER, "sources": [], "prompt_tokens": 0}

        output = self.llm.create_chat_completion(
            messages=build_messages(question, results),
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
        text = output["choices"][0]["message"]["content"].strip()
        sources = [
            {"source": r["source"], "page_number": r["page_number"], "score": r["score"]}
            for r in results
        ]
        return {
            "answer": text,
            "sources": sources,
            "prompt_tokens": output["usage"]["prompt_tokens"],
        }