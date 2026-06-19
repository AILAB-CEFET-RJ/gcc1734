from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


if __package__ in (None, ""):
    # Permite executar este arquivo diretamente com:
    # python3 src/genai/push_pull/demo.py
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.genai.push_pull.agents import (
        PullInvestigationAgent,
        PushSalesMonitor,
        TraceStep,
        coordinate_final_action,
    )
else:
    from .agents import (
        PullInvestigationAgent,
        PushSalesMonitor,
        TraceStep,
        coordinate_final_action,
    )


def print_trace(trace: list[TraceStep]) -> None:
    """Imprime o trace didático da fase Push."""
    for index, step in enumerate(trace, start=1):
        print(f"{index}. {step.kind}:")
        print_value(step.content)


def print_value(value: Any) -> None:
    """Formata valores simples para manter a saída legível em sala."""
    if isinstance(value, dict):
        for key, item in value.items():
            print(f"   {key}: {item}")
        return
    print(f"   {value}")


def default_log_path() -> Path:
    """Local padrão do log que registra prompts e respostas do LLM."""
    return Path(__file__).resolve().parent / "push_pull_demo_log.md"


def write_llm_log(log: list[dict[str, Any]], path: Path | None = None) -> Path:
    """Grava o log didático das chamadas ao LLM.

    A saída do terminal mostra o fluxo do agente. Este arquivo mostra a camada
    que normalmente ficaria escondida: prompts, respostas estruturadas e erros
    de chamada ao modelo.
    """
    target = path or default_log_path()
    parts = ["# Push-Pull Demo Log\n"]
    for index, entry in enumerate(log, start=1):
        parts.append(f"## Call {index}: {entry.get('stage', 'unknown')}\n")
        parts.append("```json\n")
        parts.append(json.dumps(entry, indent=2, ensure_ascii=False))
        parts.append("\n```\n")
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


def run_demo() -> None:
    # O mesmo log é compartilhado pelas três fases. Assim, depois da execução,
    # é possível ver a sequência completa de chamadas ao LLM.
    llm_log: list[dict[str, Any]] = []

    print("=== Fase 1: detecção com agente Push ===")
    # Nas notas, esta fase começa por um gatilho agendado às 02:00.
    # Aqui simulamos esse gatilho chamando run_scheduled_check diretamente.
    push_agent = PushSalesMonitor(llm_log=llm_log)
    push_result = push_agent.run_scheduled_check(current_time="02:00")
    print_trace(push_result.trace)
    print("\nAlerta enviado:", push_result.triggered)
    for alert in push_result.alerts:
        print_value(alert)

    print("\n=== Fase 2: investigação com agente Pull ===")
    # Agora o controle passa para o usuário. Cada item da lista representa uma
    # pergunta feita pelo analista depois de receber o alerta.
    pull_agent = PullInvestigationAgent(llm_log=llm_log)
    questions = [
        "Mostre as vendas do produto X por hora nos últimos 7 dias.",
        "Quais outros produtos cairam no mesmo período?",
        "Ha ocorrências parecidas no histórico?",
    ]
    for question in questions:
        # Em cada pergunta, o agente Pull pede ao LLM um plano, valida esse
        # plano, executa uma ferramenta e sintetiza a resposta.
        print("\nPergunta do analista:")
        print(f"   {question}")
        result = pull_agent.answer(question)
        print("ToolExecution:")
        print(f"   {result['action']}")
        print("Reasoning:")
        print(f"   {result['reasoning']}")
        print("Resposta:")
        print(f"   {result['answer']}")

    print("\nEstado de sessão do Pull:")
    # O estado de sessão mostra por que o modo Pull é conversacional:
    # perguntas posteriores podem depender do contexto já acumulado.
    print_value(pull_agent.session_state)

    print("\n=== Fase 3: ação coordenada em modo Push ===")
    # Encerrada a investigação, o sistema volta a agir de forma proativa:
    # cria um pedido e notifica o responsável.
    for action_result in coordinate_final_action(log=llm_log):
        print_value(action_result)

    # O log é gravado no fim para não poluir a saída principal do terminal.
    log_path = write_llm_log(llm_log)
    print("\nLog das chamadas ao LLM:")
    print(f"   {log_path}")


if __name__ == "__main__":
    run_demo()
