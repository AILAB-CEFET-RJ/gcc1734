from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> Path | None:
    """Carrega variáveis de ambiente a partir de locais comuns do projeto."""
    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / "src/genai/.env",
        Path(__file__).resolve().parent / ".env",
    ]

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return env_path

    load_dotenv(override=False)
    return None


def get_llm(model_backend: str = "openai", model_name: str | None = None):
    """Retorna um chat model configurado para OpenAI ou Ollama."""
    load_env()

    if model_backend == "openai":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY nao encontrada. Configure src/genai/.env ou exporte a variavel no ambiente."
            )
        return ChatOpenAI(
            temperature=0,
            model=model_name or "gpt-4o-mini",
        )

    if model_backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            temperature=0,
            model=model_name or "gemma3:latest",
        )

    raise ValueError(f"Unknown backend: {model_backend}")
