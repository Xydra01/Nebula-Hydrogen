"""Markdown document chunker starter stub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentChunk:
    header: str
    content: str
    section_path: list[str]


def chunk_markdown(text: str, max_chunk_chars: int = 500) -> list[DocumentChunk]:
    """Chunk markdown document by headers and length limits."""
    # Starter implementation: naive header split
    chunks: list[DocumentChunk] = []
    current_header = "Root"
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("#"):
            if current_lines:
                chunks.append(
                    DocumentChunk(
                        header=current_header,
                        content="\n".join(current_lines),
                        section_path=[current_header],
                    )
                )
                current_lines = []
            current_header = line.lstrip("#").strip()
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append(
            DocumentChunk(
                header=current_header,
                content="\n".join(current_lines),
                section_path=[current_header],
            )
        )
    return chunks
