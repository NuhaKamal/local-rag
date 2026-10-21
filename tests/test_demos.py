"""Every component can be run on its own; its demo must work without models."""

import importlib

import pytest

# embedder is left out on purpose: its demo loads the real nomic model
DEMOS = ["models", "config", "logs", "loader", "chunker", "index", "retriever", "generator"]


@pytest.mark.parametrize("name", DEMOS)
def test_component_demo_runs(name, capsys):
    module = importlib.import_module(f"rag_from_scratch.{name}")
    module._demo()
    assert capsys.readouterr().out.strip()  # it printed something


def test_loader_reads_the_built_in_sample_pdf(tmp_path):
    from rag_from_scratch.loader import _sample_pdf_bytes, load_pdf

    path = tmp_path / "s.pdf"
    path.write_bytes(_sample_pdf_bytes(["alpha text", "beta text"]))
    meta, pages = load_pdf(path)
    assert meta["num_pages"] == 2
    assert [p.page_number for p in pages] == [1, 2]
    assert "alpha" in pages[0].text