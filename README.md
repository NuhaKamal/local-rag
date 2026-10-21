# Build a RAG algorithm from scratch

Ask questions about your PDFs **fully offline**. No cloud and no GPU. A
retrieval-augmented generation (RAG) pipeline built by hand, one small module per
stage, with [llama.cpp](https://github.com/ggml-org/llama.cpp), TinyLlama and a
nomic embedding model.

> **This is Phase 1.** The goal is to understand every moving part: loading,
> chunking, embedding, caching, similarity search, prompting and generation, all
> written from scratch with no RAG framework. **Phase 2** (see the
> [roadmap](#roadmap)) rebuilds it with more robust components: a vector
> database, LangChain, a Streamlit interface and more.

Example session (abridged):

```
$ uv run rag ask "What does a functional movement mean?"

Question: What does a functional movement mean?

Retrieved:
  #1  p.47   score=0.79  book.pdf
  #2  p.13   score=0.79  book.pdf
  #3  p.14   score=0.78  book.pdf

Answer:
...

Sources: book.pdf p.47, book.pdf p.13, book.pdf p.14
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

## Components

Each module does one job, and **each can be run on its own** to show a sample
input and its output:

```
uv run python -m rag_from_scratch.<module>
```

| Stage | Module | Input -> Output | Idea |
|---|---|---|---|
| Shared | `models.py` | (data shapes) | `Page` and `Chunk`, the only things modules pass to each other |
| Settings | `config.py` | (constants) | Every setting in one place |
| Logging | `logs.py` | -> a configured logger | Screen output at the chosen level, a full log in `.cache/rag.log` |
| Load | `loader.py` | PDF -> `Page`s + metadata | `pypdf` text per page, with file name and page number |
| Chunk | `chunker.py` | `Page`s -> `Chunk`s | Fixed-size windows with overlap; chunks never cross pages |
| Embed | `embedder.py` | text -> 768 numbers | nomic vectors, cached by hash of (model, text) |
| Index | `index.py` | chunks + vectors <-> JSON file | Saved together, with the embedding model name |
| Retrieve | `retriever.py` | question -> `[(Chunk, score)]` | Cosine similarity with `torch`, top-K |
| Generate | `generator.py` | question + hits -> answer | Grounded prompt, score guard, TinyLlama via llama.cpp |
| CLI | `cli.py` | commands -> everything | The only module that knows the order of the stages |

The modules are decoupled: they exchange only plain data (`Page`, `Chunk`,
vectors, scores). The retriever accepts any object with an `.embed_query()`
method, so it runs and is tested without a model. Only `cli.py` wires the stages
together.

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you)
- About 1 GB of disk space for the models
- A text-based PDF (scanned PDFs need OCR, which is not supported)

## Quick start

```bash
git clone https://github.com/NuhaKamal/rag-from-scratch.git
cd rag-from-scratch
uv sync

# download the two models into models/
uv run hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir models
uv run hf download nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q4_K_M.gguf --local-dir models

# add a PDF, then build the index and ask
cp your-file.pdf data/        # Windows: Copy-Item your-file.pdf data\
uv run rag ingest
uv run rag ask "your question here"
```

The first `ingest` is slow because every chunk is embedded. Vectors are cached, so
later runs are fast.

## Usage

| Command | What it does |
|---|---|
| `rag ingest` | Index every PDF in `data/` |
| `rag ingest a.pdf b.pdf` | Index specific PDFs (this **replaces** the previous index) |
| `rag ingest --chunk-size 400 --overlap 50` | Change chunking |
| `rag ask "question"` | Retrieve chunks and answer |
| `rag ask "question" -k 5` | Retrieve 5 chunks instead of the default |
| `rag ask "question" --no-llm` | Show retrieved chunks only (fast) |
| `rag ask "question" --show-prompt` | Print the exact prompt sent to the model |
| `rag ... -v` / `-vv` | Show progress messages / every detail |

## Logging

The answer is printed; what the program is *doing* is logged. By default only
warnings reach the screen. Use `-v` for progress and `-vv` for every detail. The
file `.cache/rag.log` always records everything, with timestamps, so you can
inspect a run afterwards.

## Configuration

Edit `src/rag_from_scratch/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` / `OVERLAP` | 800 / 100 | Characters per chunk / shared between neighbours |
| `TOP_K` | 3 | Chunks given to the model |
| `MIN_SCORE` | 0.5 | If the best chunk scores lower, the model is skipped and the answer is "I don't know" |
| `N_CTX`, `MAX_NEW_TOKENS`, `TEMPERATURE` | 2048, 256, 0.1 | Generation limits and randomness |
| `LOG_LEVEL`, `LOG_PATH` | INFO, `.cache/rag.log` | Logging defaults |

## Project structure

```
rag-from-scratch/
├── pyproject.toml
├── uv.lock
├── data/                  your PDFs (git-ignored)
├── models/                GGUF models (git-ignored)
├── .cache/                embedding cache, index and log (git-ignored)
├── src/rag_from_scratch/
│   ├── models.py          shared data shapes
│   ├── config.py          settings
│   ├── logs.py            logging setup
│   ├── loader.py          PDF -> pages
│   ├── chunker.py         pages -> chunks
│   ├── embedder.py        text -> vectors (+ cache)
│   ├── index.py           save / load the index
│   ├── retriever.py       cosine similarity, top-K
│   ├── generator.py       prompt + TinyLlama
│   └── cli.py             the `rag` command
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
- **Decoupled stages.** Modules share data shapes, not each other's internals, so
  any stage can be swapped (a different PDF reader, a vector database) by
  changing one file.

## Limitations

- TinyLlama is small: answers can be sloppy, it may ignore the citation
  instruction, and it sometimes repeats the context instead of answering. Check
  the retrieved chunks and sources.
- `pypdf` flattens tables and may include headers and footers in the text.
- Scanned PDFs are not supported.
- The index is one JSON file and the search compares the question with every
  chunk: fine for hundreds to a few thousand chunks, not for millions.
- On Linux, PyPI's `torch` is a large CUDA build.

## Roadmap

**Phase 1 (this repository):** a RAG pipeline from scratch, to learn how every
part works.

**Phase 2 (next):** a more robust local RAG built from stronger components. These
are plans, not features yet:

- A **vector database** (for example Chroma or Qdrant) instead of a JSON file and
  a full scan
- **LangChain** to orchestrate the pipeline, compared against the hand-built one
- A **Streamlit** chat interface
- Better chunking (sentence or semantic), and re-ranking of retrieved chunks
- A stronger local language model
- Measuring answer quality with a small evaluation set
- Better PDF parsing (tables, scanned pages with OCR)

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
