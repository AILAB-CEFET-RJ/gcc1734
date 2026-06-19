from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import unicodedata

from .sales_data import (
    HISTORICAL_HOURLY_AVERAGE_BY_PRODUCT,
    PRODUCT,
    format_brl,
    percent_drop,
)
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
    """Um passo do trace didático exibido no terminal.

    Args:
        kind: Tipo do passo, como `Trigger`, `Check`, `Decision`,
            `ToolExecution`, `ToolOutput` ou `LLM`.
        content: Texto ou dicionário com o conteúdo do passo.
    """

    kind: str
    content: str | dict[str, Any]


@dataclass
class PushResult:
    """Resultado de uma execução do agente Push.

    Args:
        triggered: Indica se a execução gerou alerta.
        trace: Sequência de passos (usada para explicar a execução).
        alerts: Alertas enviados durante a execução.
    """

    triggered: bool
    trace: list[TraceStep]
    alerts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ProductMonitorConfig:
    """Configuração de monitoramento de um produto.

    Em produção, configurações como esta viriam de banco, API ou serviço de
    configuração. Separá-las da lógica do agente evita deixar o produto e o
    limiar hardcoded no monitor.

    Args:
        product: Código do produto monitorado.
        threshold: Limiar mínimo de queda para considerar um alerta.
        consecutive_hours: Tamanho da janela recente observada.
        historical_average: Média histórica horária esperada para o produto.
        alert_recipient: Canal ou destinatário do alerta.
    """

    product: str
    threshold: float
    consecutive_hours: int
    historical_average: float
    alert_recipient: str = "time-vendas@empresa.com"


def default_monitor_configs() -> list[ProductMonitorConfig]:
    """Retorna a política de monitoramento simulada do exemplo."""

    return [
        ProductMonitorConfig(
            product=PRODUCT,
            threshold=0.30,
            consecutive_hours=2,
            historical_average=HISTORICAL_HOURLY_AVERAGE_BY_PRODUCT[PRODUCT],
        ),
        ProductMonitorConfig(
            product="Y",
            threshold=0.20,
            consecutive_hours=2,
            historical_average=HISTORICAL_HOURLY_AVERAGE_BY_PRODUCT["Y"],
        ),
        ProductMonitorConfig(
            product="Z",
            threshold=0.25,
            consecutive_hours=2,
            historical_average=HISTORICAL_HOURLY_AVERAGE_BY_PRODUCT["Z"],
        ),
    ]


@dataclass
class PushSalesMonitor:
    """Agente Push responsável pela detecção proativa.

    Este agente representa a fase 1 das notas: um processo agendado observa
    vendas, investiga uma causa provável e envia um alerta sem esperar uma
    pergunta humana.

    Args:
        configs: Política de monitoramento com produtos, janelas e limiares.
        last_alert_keys: Chaves dos últimos alertas enviados, usadas para evitar
            duplicidade na mesma janela.
        llm_log: Lista compartilhada para registrar chamadas ao LLM.
    """

    configs: list[ProductMonitorConfig] = field(default_factory=default_monitor_configs)
    last_alert_keys: set[str] = field(default_factory=set)
    llm_log: list[dict[str, Any]] = field(default_factory=list)

    def run_scheduled_check(self, current_time: str = "02:00") -> PushResult:
        """Executa uma verificação agendada de vendas.

        Args:
            current_time: Horário simbólico do disparo. No exemplo, ele compõe
                a chave usada para evitar alertas duplicados.

        Returns:
            `PushResult` com trace completo e, se disparado, o alerta enviado.
        """

        trace: list[TraceStep] = [
            TraceStep(
                "Trigger",
                (
                    "Gatilho agendado recebido. O monitor consultará as vendas "
                    "recentes dos produtos configurados e comparará com suas médias históricas."
                ),
            )
        ]
        alerts: list[dict[str, Any]] = []

        for config in self.configs:
            trace.append(
                TraceStep(
                    "Check",
                    (
                        f"Produto {config.product} selecionado para verificação. "
                        f"Limiar: {config.threshold:.0%}. Janela: "
                        f"{config.consecutive_hours}h."
                    ),
                )
            )
            sales = query_sales(config.product, last_hours=config.consecutive_hours)
            trace.append(
                TraceStep(
                    "ToolExecution",
                    (
                        f"query_sales(product={config.product!r}, "
                        f"last_hours={config.consecutive_hours})"
                    ),
                )
            )
            trace.append(TraceStep("ToolOutput", sales))

            drop = percent_drop(sales["media_observada"], config.historical_average)
            alert_key = f"{config.product}:{current_time}:{round(drop, 3)}"
            if drop < config.threshold:
                trace.append(
                    TraceStep(
                        "Decision",
                        (
                            f"A queda de {config.product} foi de {drop:.0%}, "
                            f"abaixo do limiar de {config.threshold:.0%}. "
                            "Alerta não disparado."
                        ),
                    )
                )
                continue

            if alert_key in self.last_alert_keys:
                trace.append(
                    TraceStep(
                        "Decision",
                        (
                            f"O alerta do produto {config.product} ja foi enviado "
                            "nesta janela. Alerta duplicado suprimido."
                        ),
                    )
                )
                continue

            trace.append(
                TraceStep(
                    "Decision",
                    (
                        f"A queda media de {config.product} foi de {drop:.0%}, "
                        "acima do limiar. Verificação de causa provável habilitada."
                    ),
                )
            )
            stock = check_stock(config.product)
            trace.append(
                TraceStep("ToolExecution", f"check_stock(product={config.product!r})")
            )
            trace.append(TraceStep("ToolOutput", stock))

            alert_plan = draft_push_alert(
                sales=sales,
                stock=stock,
                drop=drop,
                threshold=config.threshold,
                log=self.llm_log,
            )
            trace.append(
                TraceStep(
                    "LLM",
                    {
                        "stage": "draft_push_alert",
                        "product": config.product,
                        "should_alert": alert_plan["should_alert"],
                        "severity": alert_plan["severity"],
                        "reasoning": alert_plan["reasoning"],
                    },
                )
            )
            if not alert_plan["should_alert"]:
                trace.append(
                    TraceStep(
                        "Decision",
                        "O LLM julgou que a observação ainda não merece alerta.",
                    )
                )
                continue

            alert_result = send_alert(config.alert_recipient, alert_plan["message"])
            trace.append(
                TraceStep(
                    "ToolExecution",
                    f"send_alert(recipient={config.alert_recipient!r}, message=...)",
                )
            )
            trace.append(TraceStep("ToolOutput", alert_result))
            self.last_alert_keys.add(alert_key)
            alerts.append(alert_result["alerta"])

        return PushResult(triggered=bool(alerts), trace=trace, alerts=alerts)


@dataclass
class PullInvestigationAgent:
    """Agente Pull responsável pela investigação conduzida pelo analista.

    Este agente representa a fase 2 das notas. Ele não roda sozinho: cada
    execução começa com uma pergunta humana. O agente usa o LLM para planejar a
    ferramenta, valida o plano, executa a ferramenta e sintetiza a resposta.

    Args:
        session_state: Estado conversacional acumulado durante a investigação.
        llm_log: Lista compartilhada para registrar chamadas ao LLM.
    """

    session_state: dict[str, Any] = field(default_factory=dict)
    llm_log: list[dict[str, Any]] = field(default_factory=list)

    def answer(self, question: str) -> dict[str, Any]:
        """Responde a uma pergunta do analista.

        Args:
            question: Pergunta feita pelo humano na fase Pull.

        Returns:
            Dicionário com modo, ação executada, observação da ferramenta,
            resposta final e justificativa do plano.
        """

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
    """Executa a fase 3: ação coordenada em modo Push.

    Depois da investigação Pull, o sistema volta a agir proativamente: cria um
    pedido de reabastecimento e notifica o gestor.

    Args:
        log: Lista opcional para registrar a chamada ao LLM que redige a
            notificação final.

    Returns:
        Lista com os resultados das ações executadas.
    """

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
    """Renderiza uma chamada de ferramenta para exibição no terminal.

    Args:
        tool_name: Nome da ferramenta executada.
        arguments: Argumentos usados na chamada.

    Returns:
        String no formato `tool(arg=value, ...)`.
    """

    args = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{tool_name}({args})"


def normalize_pull_tool_plan(question: str, tool_plan: dict[str, Any]) -> dict[str, Any]:
    """Valida e ajusta o plano produzido pelo LLM no modo Pull.

    O LLM sugere a ferramenta e os argumentos, mas o host mantém o controle do
    contrato das ferramentas. Esta função corrige planos incompatíveis com a
    base didática e explicita o ajuste no campo `reasoning`.

    Args:
        question: Pergunta original do analista.
        tool_plan: Plano retornado por `plan_pull_tool_call`.

    Returns:
        Plano possivelmente ajustado, ainda no formato `tool`, `arguments` e
        `reasoning`.
    """

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
    """Normaliza texto para comparação simples em português.

    Args:
        text: Texto com ou sem acentos.

    Returns:
        Texto em minúsculas e sem acentos.
    """

    without_accents = unicodedata.normalize("NFKD", text)
    ascii_text = without_accents.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()
