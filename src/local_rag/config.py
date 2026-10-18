"""Project-wide settings. We add more here as the project grows."""


from pathlib import Path

# --- retrieval ---
TOP_K = 3          # chunks handed to the LLM (TinyLlama's context is small)
MIN_SCORE = 0.6    # best chunk must score at least this, else we refuse (TUNE THIS)

# --- generation ---
LLM_MODEL_PATH = Path("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
N_CTX = 2048           # context window in tokens (prompt + answer)
MAX_NEW_TOKENS = 256   # room reserved for the answer
TEMPERATURE = 0.1      # low = focused, repeatable