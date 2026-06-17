from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def default_persist_directory() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "rag_chroma_store"


def build_embedding_model(model: str = "text-embedding-3-small"):
    return OpenAIEmbeddings(model=model)


def build_vectorstore(chunks, persist_directory: str | Path | None = None, embedding_model=None):
    persist_path = Path(persist_directory) if persist_directory else default_persist_directory()
    persist_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = embedding_model or build_embedding_model()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )
