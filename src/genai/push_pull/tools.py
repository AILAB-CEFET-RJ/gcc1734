from __future__ import annotations

from statistics import mean
from typing import Any

from .sales_data import (
    HISTORICAL_HOURLY_AVERAGE,
    HOURLY_SALES,
    INCIDENT_HISTORY,
    STOCK_BY_PRODUCT,
    format_brl,
)


OUTBOX: list[dict[str, Any]] = []
RESTOCK_ORDERS: list[dict[str, Any]] = []


def query_sales(
    product: str | None = None,
    *,
    last_hours: int | None = None,
    compare_all: bool = False,
) -> dict[str, Any]:
    rows = HOURLY_SALES
    if product is not None:
        rows = [row for row in rows if row.product == product]

    rows = sorted(rows, key=lambda row: row.hour)
    if last_hours is not None:
        rows = rows[:last_hours]

    if compare_all:
        products = sorted({row.product for row in HOURLY_SALES})
        comparison = {}
        for item in products:
            item_rows = [row for row in HOURLY_SALES if row.product == item]
            avg = mean(row.revenue for row in item_rows)
            comparison[item] = {
                "media_periodo": avg,
                "media_periodo_fmt": format_brl(avg),
                "queda_relevante": item == "X",
            }
        return {"comparacao": comparison}

    average = mean(row.revenue for row in rows) if rows else 0.0
    return {
        "produto": product,
        "media_historica_hora": HISTORICAL_HOURLY_AVERAGE,
        "media_historica_hora_fmt": format_brl(HISTORICAL_HOURLY_AVERAGE),
        "media_observada": average,
        "media_observada_fmt": format_brl(average),
        "vendas": [
            {
                "hora": row.hour,
                "faturamento": row.revenue,
                "faturamento_fmt": format_brl(row.revenue),
                "regiao": row.region,
            }
            for row in rows
        ],
    }


def check_stock(product: str) -> dict[str, Any]:
    return {"produto": product, **STOCK_BY_PRODUCT[product]}


def query_history(product: str, cause: str) -> dict[str, Any]:
    incidents = [
        incident
        for incident in INCIDENT_HISTORY
        if incident["produto"] == product and incident["causa"] == cause
    ]
    avg_recovery = mean(
        incident["tempo_recuperacao_horas"] for incident in incidents
    ) if incidents else None
    return {
        "produto": product,
        "causa": cause,
        "ocorrencias": incidents,
        "tempo_medio_recuperacao_horas": avg_recovery,
    }


def send_alert(recipient: str, message: str) -> dict[str, Any]:
    alert = {
        "tipo": "alerta",
        "destinatario": recipient,
        "mensagem": message,
    }
    OUTBOX.append(alert)
    return {"status": "enviado", "alerta": alert}


def create_restock_order(product: str, quantity: int, priority: str) -> dict[str, Any]:
    order = {
        "produto": product,
        "quantidade": quantity,
        "prioridade": priority,
    }
    RESTOCK_ORDERS.append(order)
    return {"status": "criado", "pedido": order}


def notify_manager(message: str) -> dict[str, Any]:
    notification = {
        "tipo": "notificacao",
        "destinatario": "gerente-comercial@empresa.com",
        "mensagem": message,
    }
    OUTBOX.append(notification)
    return {"status": "enviado", "notificacao": notification}

