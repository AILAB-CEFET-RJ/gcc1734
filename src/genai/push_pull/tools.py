from __future__ import annotations

from statistics import mean
from typing import Any

from .sales_data import (
    HISTORICAL_HOURLY_AVERAGE,
    HISTORICAL_HOURLY_AVERAGE_BY_PRODUCT,
    HOURLY_SALES,
    INCIDENT_HISTORY,
    STOCK_BY_PRODUCT,
    format_brl,
)


# Neste exemplo, as ferramentas são funções Python comuns.
# Em produção, esta camada seria o ponto de contato com banco de dados,
# ERP, serviços HTTP, filas, e-mail ou MCP Servers.
OUTBOX: list[dict[str, Any]] = []
RESTOCK_ORDERS: list[dict[str, Any]] = []


def query_sales(
    product: str | None = None,
    *,
    last_hours: int | None = None,
    compare_all: bool = False,
) -> dict[str, Any]:
    """Consulta vendas simuladas.

    Esta é uma ferramenta de leitura: observa o estado comercial sem causar
    efeitos colaterais. O agente Push a usa para detectar anomalias; o agente
    Pull usa a mesma ferramenta para responder perguntas do analista.

    Args:
        product: Código do produto a consultar. Quando `None`, considera todos
            os produtos.
        last_hours: Quantidade de registros horários recentes a considerar.
            Quando `None`, usa toda a janela simulada disponível.
        compare_all: Quando `True`, ignora o filtro de produto e devolve uma
            comparação agregada entre produtos.

    Returns:
        Dicionário com vendas observadas, médias formatadas e, quando
        `compare_all=True`, comparação entre produtos.
    """
    rows = HOURLY_SALES
    if product is not None:
        rows = [row for row in rows if row.product == product]

    rows = sorted(rows, key=lambda row: row.hour)
    if last_hours is not None:
        rows = rows[:last_hours]

    if compare_all:
        # Modo usado na investigação Pull: comparar produtos ajuda a distinguir
        # uma queda localizada de um problema sistêmico.
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
    historical_average = (
        HISTORICAL_HOURLY_AVERAGE_BY_PRODUCT.get(product, HISTORICAL_HOURLY_AVERAGE)
        if product is not None
        else HISTORICAL_HOURLY_AVERAGE
    )
    return {
        "produto": product,
        "media_historica_hora": historical_average,
        "media_historica_hora_fmt": format_brl(historical_average),
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
    """Consulta o estoque simulado de um produto.

    Esta ferramenta não decide se há alerta. Ela apenas devolve uma observação
    que o agente pode usar para montar o diagnóstico.

    Args:
        product: Código do produto a consultar.

    Returns:
        Dicionário com produto, quantidade em estoque e último reabastecimento.
    """
    return {"produto": product, **STOCK_BY_PRODUCT[product]}


def query_history(product: str, cause: str) -> dict[str, Any]:
    """Recupera ocorrências passadas de um produto.

    Esta ferramenta faz o papel de memória operacional do domínio. No modo Pull,
    ela apoia perguntas de seguimento como "isso já aconteceu antes?".

    Args:
        product: Código do produto investigado.
        cause: Causa ou filtro do incidente, por exemplo `estoque_zerado`.

    Returns:
        Dicionário com ocorrências encontradas e tempo médio de recuperação.
    """
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
    """Envia um alerta simulado.

    Esta é uma ferramenta com efeito colateral: altera `OUTBOX`. No mundo real,
    aqui haveria uma chamada para e-mail, Slack, Teams, PagerDuty ou outro canal
    de notificação.

    Args:
        recipient: Destinatário do alerta.
        message: Conteúdo da mensagem enviada.

    Returns:
        Dicionário com status de envio e o alerta registrado.
    """
    alert = {
        "tipo": "alerta",
        "destinatario": recipient,
        "mensagem": message,
    }
    OUTBOX.append(alert)
    return {"status": "enviado", "alerta": alert}


def create_restock_order(product: str, quantity: int, priority: str) -> dict[str, Any]:
    """Cria um pedido de reabastecimento simulado.

    Esta função representa uma ação operacional da fase final. No exemplo,
    apenas registra o pedido em memória; em produção, chamaria um ERP, sistema
    de compras ou workflow com aprovação humana.

    Args:
        product: Código do produto a reabastecer.
        quantity: Quantidade solicitada.
        priority: Prioridade operacional do pedido.

    Returns:
        Dicionário com status de criação e dados do pedido.
    """
    order = {
        "produto": product,
        "quantidade": quantity,
        "prioridade": priority,
    }
    RESTOCK_ORDERS.append(order)
    return {"status": "criado", "pedido": order}


def notify_manager(message: str) -> dict[str, Any]:
    """Notifica o gestor responsável.

    Esta é a segunda ação Push da fase final: depois da investigação, o sistema
    comunica uma resposta coordenada ao responsável.

    Args:
        message: Mensagem executiva a ser enviada ao gestor.

    Returns:
        Dicionário com status de envio e a notificação registrada.
    """
    notification = {
        "tipo": "notificacao",
        "destinatario": "gerente-comercial@empresa.com",
        "mensagem": message,
    }
    OUTBOX.append(notification)
    return {"status": "enviado", "notificacao": notification}
