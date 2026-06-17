from __future__ import annotations

from pathlib import Path


SAMPLE_TEXT = """
Databases are organized collections of structured information or data, typically stored electronically in a computer system.
Artificial intelligence (AI) refers to systems that can perform tasks that normally require human intelligence, such as reasoning and learning.
LLM-based agents combine large language models with external tools.
""".strip()


def default_sample_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "rag_sample_doc.txt"


def ensure_sample_document(path: Path | None = None) -> Path:
    target = path or default_sample_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SAMPLE_TEXT + "\n", encoding="utf-8")
    return target
