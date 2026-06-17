from __future__ import annotations

import json
import re
from typing import Any


MONTHLY_REVENUE = {
    ("Sul", "2024-11"): {
        "faturamento_total": 847293.50,
        "moeda": "BRL",
        "periodo": "2024-11",
        "registros": 1284,
    },
    ("Sudeste", "2024-11"): {
        "faturamento_total": 1324088.10,
        "moeda": "BRL",
        "periodo": "2024-11",
        "registros": 2011,
    },
    ("Nordeste", "2024-11"): {
        "faturamento_total": 532901.22,
        "moeda": "BRL",
        "periodo": "2024-11",
        "registros": 944,
    },
}

REPORTS = {
    ("Sul", "2024-11"): (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]>>endobj\n"
        b"% Relatorio sintetico da regiao Sul - novembro de 2024\n"
        b"%%EOF"
    )
}

DB_SCHEMA_METADATA = {
    "table": "vendas",
    "columns": [
        "id",
        "data_venda",
        "regiao",
        "valor",
        "produto_id",
        "vendedor_id",
    ],
    "access_policy": "somente leitura",
    "server": "db-server-vendas-readonly",
}


def pretty_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def pick_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def content_to_text(content: Any) -> str:
    if hasattr(content, "text"):
        return content.text
    return str(content)


def uri_template_to_regex(uri_template: str) -> re.Pattern[str]:
    pattern = re.escape(uri_template)
    pattern = pattern.replace(r"\{period\}", r"(?P<period>[^/]+)")
    pattern = pattern.replace(r"\{region_slug\}", r"(?P<region_slug>[^/]+)")
    return re.compile(f"^{pattern}$")


def region_to_slug(region: str) -> str:
    return region.strip().lower()


def extract_sql_region(sql: str) -> str:
    match = re.search(r"regiao\s*=\s*'([^']+)'", sql, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Filtro de regiao nao encontrado no SQL: {sql}")
    return match.group(1)


def infer_period_from_sql(sql: str) -> str:
    normalized_sql = sql.upper()

    if (
        "NOW() - INTERVAL '1 MONTH'" in normalized_sql
        and "DATE_TRUNC('MONTH', NOW())" in normalized_sql
    ):
        return "2024-11"

    if (
        "CURRENT_DATE - INTERVAL '1 MONTH'" in normalized_sql
        and "DATE_TRUNC('MONTH', CURRENT_DATE)" in normalized_sql
    ):
        return "2024-11"

    explicit_match = re.search(r"periodo\s*=\s*'([^']+)'", sql, flags=re.IGNORECASE)
    if explicit_match:
        return explicit_match.group(1)

    raise ValueError(f"Nao foi possivel inferir o periodo a partir do SQL: {sql}")


def format_brl(value: float) -> str:
    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def format_int_pt_br(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def format_period_pt_br(period: str) -> str:
    months = {
        "01": "janeiro",
        "02": "fevereiro",
        "03": "marco",
        "04": "abril",
        "05": "maio",
        "06": "junho",
        "07": "julho",
        "08": "agosto",
        "09": "setembro",
        "10": "outubro",
        "11": "novembro",
        "12": "dezembro",
    }
    year, month = period.split("-")
    return f"{months[month]} de {year}"
