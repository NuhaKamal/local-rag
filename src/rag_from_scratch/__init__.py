"""
rag_from_scratch: a RAG pipeline built from scratch (Phase 1).

Each module is one stage, and each can be run on its own:

    python -m rag_from_scratch.<module>

  models     the shared data shapes (Page, Chunk)
  config     every setting in one place
  logs       one-time logging setup
  loader     PDF -> pages + metadata
  chunker    pages -> fixed-size chunks with overlap
  embedder   text -> vectors, with a disk cache
  index      save and load chunks + vectors
  retriever  question -> top-K chunks by cosine similarity
  generator  chunks + question -> prompt -> answer
  cli        wires the stages together (the `rag` command)
"""