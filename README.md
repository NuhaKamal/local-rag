# local-rag

Ask questions about your PDFs **fully offline**. No cloud and no GPU. A small
retrieval-augmented generation (RAG) tool built from scratch with
[llama.cpp](https://github.com/ggml-org/llama.cpp), TinyLlama and a nomic
embedding model.

Example session (abridged):

```
$ uv run local-rag ask "What does a functional movement mean?"

Retrieved:
  #1  p.45   score=0.71  book.pdf
  ...

Answer:
A functional movement means ... (p. 45)

Sources: book.pdf p.45, book.pdf p.12
```

## How it works

```
 INGEST (once)                                   ASK (every question)

 PDF -> pages + metadata                         question
     -> fixed-size chunks (with overlap)           -> embed (nomic)
     -> embeddings (nomic, cached on disk)         -> cosine similarity vs. all chunks
     -> index (chunks + vectors) on disk           -> top-K chunks
                                                   -> prompt (context + rules)
                                                   -> TinyLlama answer + page citations
```

| Stage | Module | Idea |
|---|---|---|
| Load | `loader.py` | `pypdf` text per page, with file name and page number |
| Chunk | `chunker.py` | Fixed-size windows with overlap; chunks never cross pages |
| Embed | `embedder.py` | nomic vectors, cached by hash of (model, text) |
| Index | `index.py` | Chunks and vectors saved together, with the embedding model name |
| Retrieve | `retrieval/retriever.py` | Cosine similarity with `torch`, top-K |
| Generate | `generation/generator.py` | Grounded prompt, score guard, TinyLlama via llama.cpp |
| CLI | `cli.py` | `ingest` and `ask` commands |

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you)
- About 1 GB of disk space for the models
- A text-based PDF (scanned PDFs need OCR, which is not supported)

## Quick start

```bash
git clone https://github.com/NuhaKamal/local-rag.git
cd local-rag
uv sync

# download the two models into models/
uv run hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir models
uv run hf download nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q4_K_M.gguf --local-dir models

# add a PDF, then build the index and ask
cp your-file.pdf data/        # Windows: Copy-Item your-file.pdf data\
uv run local-rag ingest
uv run local-rag ask "your question here"
```

The first `ingest` is slow because every chunk is embedded. Vectors are cached, so
later runs are fast.

## Usage

| Command | What it does |
|---|---|
| `local-rag ingest` | Index every PDF in `data/` |
| `local-rag ingest a.pdf b.pdf` | Index specific PDFs (this **replaces** the previous index) |
| `local-rag ingest --chunk-size 400 --overlap 50` | Change chunking |
| `local-rag ask "question"` | Retrieve chunks and answer |
| `local-rag ask "question" -k 5` | Retrieve 5 chunks instead of the default |
| `local-rag ask "question" --no-llm` | Show retrieved chunks only (fast) |
| `local-rag ask "question" --show-prompt` | Print the exact prompt sent to the model |

## Configuration

Edit `src/local_rag/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` / `OVERLAP` | 800 / 100 | Characters per chunk / shared between neighbours |
| `TOP_K` | 3 | Chunks given to the model |
| `MIN_SCORE` | tuned per document | If the best chunk scores lower, the model is skipped and the answer is "I don't know" |
| `N_CTX`, `MAX_NEW_TOKENS`, `TEMPERATURE` | 2048, 256, 0.1 | Generation limits and randomness |

## Project structure

```
local-rag/
├── pyproject.toml
├── uv.lock
├── data/                  your PDFs (git-ignored)
├── models/                GGUF models (git-ignored)
├── .cache/                embedding cache and index (git-ignored)
├── src/local_rag/
│   ├── cli.py
│   ├── config.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── index.py
│   ├── retrieval/retriever.py
│   └── generation/generator.py
└── tests/
```

## Design notes

- **Two models.** TinyLlama writes answers; a dedicated embedding model does
  the searching. Chat models make poor search embeddings.
- **Cosine similarity, not Euclidean distance.** It compares direction (meaning)
  and ignores vector length.
- **Cache vs. index.** The cache avoids recomputing embeddings. The index is the
  searchable knowledge base that `ask` reads.
- **Guard against unanswerable questions.** Retrieval always returns K chunks, so
  a minimum score stops the model from improvising on irrelevant text.

## Limitations

- TinyLlama is small: answers can be sloppy, and it may ignore the citation
  instruction. Check the retrieved chunks and sources.
- `pypdf` flattens tables and may include headers and footers in the text.
- Scanned PDFs are not supported.
- The index is one JSON file: fine for hundreds to a few thousand chunks, not for
  millions.
- On Linux, PyPI's `torch` is a large CUDA build.

## Tests

```bash
uv run pytest
```

The tests use small fake inputs and do not need the models.

## Built with

[llama.cpp](https://github.com/ggml-org/llama.cpp) ·
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) ·
[TinyLlama](https://github.com/jzhang38/TinyLlama) ·
[nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) ·
[PyTorch](https://pytorch.org) · [pypdf](https://pypdf.readthedocs.io) · [uv](https://docs.astral.sh/uv/)