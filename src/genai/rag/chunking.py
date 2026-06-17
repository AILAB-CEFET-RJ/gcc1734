from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_recursive_splitter(chunk_size: int = 80, chunk_overlap: int = 20):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def split_documents(documents, chunk_size: int = 80, chunk_overlap: int = 20):
    splitter = build_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return splitter, chunks
