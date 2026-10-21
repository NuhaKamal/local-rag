"""
config.py: every tunable setting in one place.

Change a value here instead of hunting through the code.

RUN IT ON ITS OWN:
  uv run python -m rag_from_scratch.config
"""

from pathlib import Path

# --- ingestion ---
CHUNK_SIZE = 800    # characters per chunk
OVERLAP = 100       # characters shared between neighbouring chunks
INDEX_PATH = Path(".cache/index.json")   # the searchable index written by `rag ingest`

# --- retrieval ---
TOP_K = 3          # chunks handed to the LLM
MIN_SCORE = 0.5    # best chunk must score at least this, else we refuse (TUNE THIS)

# --- generation ---
LLM_MODEL_PATH = Path("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
N_CTX = 2048           # context window in tokens (prompt + answer)
MAX_NEW_TOKENS = 256   # room reserved for the answer
TEMPERATURE = 0.1      # low = focused, repeatable

# --- logging ---
LOG_PATH = Path(".cache/rag.log")   # history of what the program did
LOG_LEVEL = "INFO"                  # DEBUG (most detail), INFO, WARNING, ERROR


def _demo() -> None:
    print("== config.py: current settings ==\n")
    for name, value in list(globals().items()):
        if name.isupper():
            print(f"  {name:<16} = {value}")


if __name__ == "__main__":
    _demo()