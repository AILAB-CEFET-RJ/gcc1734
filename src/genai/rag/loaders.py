from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import TextLoader


def load_text_documents(path: str | Path):
    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()
