from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import unicodedata

from .sales_data import HISTORICAL_HOURLY_AVERAGE, PRODUCT, format_brl, percent_drop
from .llm_support import (
    draft_manager_notification,
    draft_pull_answer,
    draft_push_alert,
    plan_pull_tool_call,
)
from .tools import (
    check_stock,
    create_restock_order,
    notify_manager,
    query_history,
    query_sales,
    send_alert,
)


@dataclass
class TraceStep:
    kind: str
    content: str | dict[str, Any]


@dataclass
class PushResult:
    triggered: bool
    trace: list[TraceStep]
    alert: dict[str, Any] | None = None


@dataclass
class PushSalesMonitor:
    threshold: float = 0.30
    consecutive_hours: int = 2
    last_alert_key: str | None = None
    llm_log: list[dict[str, Any]] = field(default_factory=list)

    def run_scheduled_check(self, current_time: str = "02:00") -> PushResult:
        trace: list[TraceStep] = [
            TraceStep(
                "Thought",
                (
                    "Meu ciclo agendado foi disparado. Vou consultar as vendas "
                    "recentes e comparar com a media historica."
                ),
            )
        ]

        sales = query_sales(PRODUCT, last_hours=self.consecutive_hours)
        trace.append(
            TraceStep(
                "Action",
                f"query_sales(product={PRODUCT!r}, last_hours={self.consecutive_hours})",
            )
        )
        trace.append(TraceStep("Observation", sales))

        drop = percent_drop(sales["media_observada"], HISTORICAL_HOURLY_AVERAGE)
        alert_key = f"{PRODUCT}:{current_time}:{round(drop, 3)}"
        if drop < self.threshold:
            trace.append(
                TraceStep(
                    "Thought",
                    (
                        f"A queda foi de {drop:.0%}, abaixo do limiar de "
                        f"{self.threshold:.0%}. Nao vou alertar."
                    ),
                )
            )
            return PushResult(triggered=False, trace=trace)

        if self.last_alert_key == alert_key:
            trace.append(
                TraceStep(
                    "Thought",
                    "Este alerta ja foi enviado nesta janela. Vou evitar duplicidade.",
                )
            )
            return PushResult(triggered=False, trace=trace)

        trace.append(
            TraceStep(
                "Thought",
                (
                    f"A queda media foi de {drop:.0%}, acima do limiar. "
                    "Vou verificar uma causa provavel antes de alertar."
                ),
            )
        )
        stock = check_stock(PRODUCT)
        trace.append(TraceStep("Action", f"check_stock(product={PRODUCT!r})"))
        trace.append(TraceStep("Observation", stock))

        alert_plan = draft_push_alert(
            sales=sales,
            stock=stock,
            drop=drop,
            threshold=self.threshold,
            log=self.llm_log,
        )
        trace.append(
            TraceStep(
                "LLM",
                {
                    "stage": "draft_push_alert",
                    "should_alert": alert_plan["should_alert"],
                    "severity": alert_plan["severity"],
                    "reasoning": alert_plan["reasoning"],
                },
            )
        )
        if not alert_plan["should_alert"]:
            trace.append(
                TraceStep(
                    "Thought",
                    "O LLM julgou que a observação ainda não merece alerta.",
                )
            )
            return PushResult(triggered=False, trace=trace)

        message = alert_plan["message"]
        alert_result = send_alert("time-vendas@empresa.com", message)
        trace.append(
            TraceStep(
                "Action",
                "send_alert(recipient='time-vendas@empresa.com', message=...)",
            )
        )
        trace.append(TraceStep("Observation", alert_result))
        self.last_alert_key = alert_key
        return PushResult(triggered=True, trace=trace, alert=alert_result["alerta"])


@dataclass
class PullInvestigationAgent:
    session_state: dict[str, Any] = field(default_factory=dict)
    llm_log: list[dict[str, Any]] = field(default_factory=list)

    def answer(self, question: str) -> dict[str, Any]:
        self.session_state.setdefault("perguntas", []).append(question)
        tool_plan = plan_pull_tool_call(question, self.session_state, self.llm_log)
        tool_plan = normalize_pull_tool_plan(question, tool_plan)
        tool_name = tool_plan["tool"]
        arguments = tool_plan["arguments"]

        if tool_name == "query_sales":
            product = arguments.get("product")
            last_hours = arguments.get("last_hours")
            compare_all = bool(arguments.get("compare_all", False))
            if product == "X" or product is None:
                self.session_state["produto_em_foco"] = PRODUCT
            self.session_state.setdefault("janela_em_foco", "2024-11-18 00:00-08:00")
            observation = query_sales(
                product=product,
                last_hours=last_hours,
                compare_all=compare_all,
            )
            answer = draft_pull_answer(
                question,
                tool_plan,
                observation,
                self.session_state,
                self.llm_log,
            )
            return {
                "modo": "Pull",
                "action": render_action(tool_name, arguments),
                "observation": observation,
                "answer": answer["answer"],
                "reasoning": tool_plan["reasoning"],
            }

        if tool_name == "query_history":
            product = arguments.get("product") or PRODUCT
            cause = arguments.get("cause") or "estoque_zerado"
            self.session_state["produto_em_foco"] = PRODUCT
            observation = query_history(product, cause)
            answer = draft_pull_answer(
                question,
                tool_plan,
                observation,
                self.session_state,
                self.llm_log,
            )
            return {
                "modo": "Pull",
                "action": render_action(tool_name, {"product": product, "cause": cause}),
                "observation": observation,
                "answer": answer["answer"],
                "reasoning": tool_plan["reasoning"],
            }

        return {
            "modo": "Pull",
            "action": "clarify()",
            "observation": {"session_state": self.session_state},
            "answer": "Preciso de uma pergunta mais específica para investigar o alerta.",
            "reasoning": tool_plan["reasoning"],
        }


def coordinate_final_action(
    log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    order = create_restock_order(PRODUCT, quantity=500, priority="urgente")
    notification_plan = draft_manager_notification(
        {
            "pedido": order["pedido"],
            "diagnostico": "Produto X com estoque zerado e queda relevante de vendas.",
            "impacto_estimado": format_brl(34_320.0),
        },
        log,
    )
    notification = notify_manager(notification_plan["message"])
    return [order, notification]


def render_action(tool_name: str, arguments: dict[str, Any]) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{tool_name}({args})"


def normalize_pull_tool_plan(question: str, tool_plan: dict[str, Any]) -> dict[str, Any]:
    normalized_question = normalize_text(question)
    adjusted = {
        "tool": tool_plan["tool"],
        "arguments": dict(tool_plan.get("arguments", {})),
        "reasoning": tool_plan["reasoning"],
    }

    if "outros produtos" in normalized_question or "mesmo periodo" in normalized_question:
        adjusted["tool"] = "query_sales"
        adjusted["arguments"] = {
            "product": None,
            "last_hours": None,
            "compare_all": True,
        }
        adjusted["reasoning"] = (
            tool_plan["reasoning"]
            + " Plano ajustado pelo host: comparação entre produtos deve usar query_sales(compare_all=True)."
        )
        return adjusted

    if "historico" in normalized_question or "ocorrencias parecidas" in normalized_question:
        adjusted["tool"] = "query_history"
        adjusted["arguments"] = {
            "product": PRODUCT,
            "cause": "estoque_zerado",
        }
        adjusted["reasoning"] = (
            tool_plan["reasoning"]
            + " Plano ajustado pelo host: o histórico relevante do alerta atual usa cause='estoque_zerado'."
        )
        return adjusted

    if "ultimos 7 dias" in normalized_question or "por hora" in normalized_question:
        adjusted["tool"] = "query_sales"
        adjusted["arguments"] = {
            "product": PRODUCT,
            "last_hours": None,
            "compare_all": False,
        }
        adjusted["reasoning"] = (
            tool_plan["reasoning"]
            + " Plano ajustado pelo host: a base didática possui a janela horária simulada do incidente."
        )
        return adjusted

    return adjusted


def normalize_text(text: str) -> str:
    without_accents = unicodedata.normalize("NFKD", text)
    ascii_text = without_accents.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()
