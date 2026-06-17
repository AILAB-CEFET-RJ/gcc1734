from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> Path | None:
    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / "src/genai/.env",
    ]

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return env_path

    load_dotenv(override=False)
    return None


def get_llm(provider: str = "openai"):
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY nao encontrada. Configure um .env ou exporte a variavel no ambiente."
            )
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model="llama3.1", temperature=0)

    raise ValueError(f"Provider desconhecido: {provider}")
