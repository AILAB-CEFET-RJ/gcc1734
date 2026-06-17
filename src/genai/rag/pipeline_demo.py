from __future__ import annotations

from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from chunking import split_documents
    from generation import build_rag_prompt, generate_answer
    from indexing import build_embedding_model, build_vectorstore, default_persist_directory
    from llm_config import get_llm, load_env
    from loaders import load_text_documents
    from retrieval import build_retriever, retrieve_context
    from sample_corpus import ensure_sample_document
else:
    from .chunking import split_documents
    from .generation import build_rag_prompt, generate_answer
    from .indexing import build_embedding_model, build_vectorstore, default_persist_directory
    from .llm_config import get_llm, load_env
    from .loaders import load_text_documents
    from .retrieval import build_retriever, retrieve_context
    from .sample_corpus import ensure_sample_document


def run_demo(provider: str = "openai") -> None:
    env_path = load_env()
    sample_path = ensure_sample_document()
    docs = load_text_documents(sample_path)
    splitter, chunks = split_documents(docs, chunk_size=80, chunk_overlap=20)
    embedding_model = build_embedding_model()
    vectorstore = build_vectorstore(chunks, default_persist_directory(), embedding_model)
    retriever = build_retriever(vectorstore, k=2)
    llm = get_llm(provider)
    prompt = build_rag_prompt()

    print("=== Ambiente ===")
    print(".env carregado de:", env_path or "nenhum arquivo encontrado")
    print("Documento:", sample_path)
    print("Chunks:", len(chunks))
    print("Persist directory:", default_persist_directory())

    print("\n=== Chunks ===")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk.page_content}")

    questions = [
        "What is a database?",
        "What are LLM-based agents?",
        "What is artificial intelligence?",
        "What is Quantum Physics?",
    ]

    for question in questions:
        docs_found, context = retrieve_context(retriever, question)
        answer = generate_answer(llm, question, context, prompt)

        print("\n" + "=" * 60)
        print("QUESTION:")
        print(question)
        print("-" * 60)
        print("RETRIEVED:")
        for doc in docs_found:
            print("-", doc.page_content)
        print("-" * 60)
        print("ANSWER:")
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    run_demo()
