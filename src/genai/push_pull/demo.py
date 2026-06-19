from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


if __package__ in (None, ""):
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
    for index, step in enumerate(trace, start=1):
        print(f"{index}. {step.kind}:")
        print_value(step.content)


def print_value(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            print(f"   {key}: {item}")
        return
    print(f"   {value}")


def default_log_path() -> Path:
    return Path(__file__).resolve().parent / "push_pull_demo_log.md"


def write_llm_log(log: list[dict[str, Any]], path: Path | None = None) -> Path:
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
    llm_log: list[dict[str, Any]] = []

    print("=== Fase 1: detecção com agente Push ===")
    push_agent = PushSalesMonitor(llm_log=llm_log)
    push_result = push_agent.run_scheduled_check(current_time="02:00")
    print_trace(push_result.trace)
    print("\nAlerta enviado:", push_result.triggered)
    if push_result.alert:
        print_value(push_result.alert)

    print("\n=== Fase 2: investigação com agente Pull ===")
    pull_agent = PullInvestigationAgent(llm_log=llm_log)
    questions = [
        "Mostre as vendas do produto X por hora nos últimos 7 dias.",
        "Quais outros produtos cairam no mesmo período?",
        "Ha ocorrências parecidas no histórico?",
    ]
    for question in questions:
        print("\nPergunta:")
        print(f"   {question}")
        result = pull_agent.answer(question)
        print("Action:")
        print(f"   {result['action']}")
        print("Reasoning:")
        print(f"   {result['reasoning']}")
        print("Resposta:")
        print(f"   {result['answer']}")

    print("\nEstado de sessão do Pull:")
    print_value(pull_agent.session_state)

    print("\n=== Fase 3: ação coordenada em modo Push ===")
    for action_result in coordinate_final_action(log=llm_log):
        print_value(action_result)

    log_path = write_llm_log(llm_log)
    print("\nLog das chamadas ao LLM:")
    print(f"   {log_path}")


if __name__ == "__main__":
    run_demo()
