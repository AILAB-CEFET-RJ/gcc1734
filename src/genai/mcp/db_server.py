from __future__ import annotations

from pathlib import Path
import sys

from fastmcp import FastMCP


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from shared_data import (
        MONTHLY_REVENUE,
        extract_sql_region,
        format_period_pt_br,
        infer_period_from_sql,
    )
else:
    from .shared_data import (
        MONTHLY_REVENUE,
        extract_sql_region,
        format_period_pt_br,
        infer_period_from_sql,
    )


db_server = FastMCP("db-server-vendas-readonly")


@db_server.tool
def query_database(sql: str) -> dict:
    """Executa consultas SQL somente-leitura sobre o banco de vendas."""
    region = extract_sql_region(sql)
    period = infer_period_from_sql(sql)
    key = (region, period)

    if key not in MONTHLY_REVENUE:
        raise ValueError(f"Nenhum dado encontrado para {region=} e {period=}.")

    return {
        "region": region,
        **MONTHLY_REVENUE[key],
    }


@db_server.prompt
def resumir_faturamento(region: str, period: str, faturamento_total: float, registros: int) -> str:
    """Template para orientar a redação de um resumo financeiro curto."""
    period_label = format_period_pt_br(period)
    return (
        "Redija uma resposta curta em portugues sobre faturamento. "
        f"Regiao: {region}. Periodo: {period_label}. "
        f"Faturamento total: {faturamento_total:.2f}. Registros: {registros}. "
        "Mencione explicitamente o periodo e o numero de registros."
    )


if __name__ == "__main__":
    db_server.run()
