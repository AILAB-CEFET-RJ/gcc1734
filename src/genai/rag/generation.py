from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_rag_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the context provided below to answer. "
                "If the context does not provide information to answer, say 'I dont know.'.\n"
                "BEGIN of CONTEXT\n{context}\nEND OF CONTEXT",
            ),
            ("human", "{question}"),
        ]
    )


def generate_answer(llm, question: str, context: str, prompt_template=None) -> str:
    prompt = prompt_template or build_rag_prompt()
    chain = prompt | llm
    return chain.invoke({"context": context, "question": question}).content
