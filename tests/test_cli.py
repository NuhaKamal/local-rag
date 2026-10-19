import pytest

from local_rag.cli import build_parser, main
from local_rag.config import CHUNK_SIZE, OVERLAP


def test_ask_collects_words_and_options():
    args = build_parser().parse_args(["ask", "what", "is", "a", "score", "-k", "5", "--no-llm"])
    assert args.question == ["what", "is", "a", "score"]
    assert args.top_k == 5
    assert args.no_llm is True


def test_ingest_defaults():
    args = build_parser().parse_args(["ingest"])
    assert args.pdfs == []
    assert args.chunk_size == CHUNK_SIZE
    assert args.overlap == OVERLAP


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_ask_without_an_index_fails_politely(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("local_rag.cli.INDEX_PATH", tmp_path / "missing.json")
    assert main(["ask", "hello"]) == 1
    assert "ingest" in capsys.readouterr().err