from __future__ import annotations

from chunker import chunk_markdown


def test_chunk_sections():
    doc = """# Introduction
Welcome to Smart Archive.

## Installation
Run pip install smart-archive.

## Architecture
Smart Archive uses hybrid retrieval.
"""
    chunks = chunk_markdown(doc)
    assert len(chunks) >= 3
    assert any("Introduction" in c.header for c in chunks)
    assert any("Installation" in c.header for c in chunks)


def test_empty_document():
    assert chunk_markdown("") == []
