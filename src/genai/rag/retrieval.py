from __future__ import annotations


def build_retriever(vectorstore, k: int = 2):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_context(retriever, question: str):
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)
    return docs, context
