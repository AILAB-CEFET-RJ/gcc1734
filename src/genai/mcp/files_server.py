from __future__ import annotations

from pathlib import Path
import sys

from fastmcp import FastMCP


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from shared_data import REPORTS
else:
    from .shared_data import REPORTS


files_server = FastMCP("files-server-relatorios")


@files_server.resource(
    "file:///relatorios/vendas/{period}/{region_slug}.pdf",
    mime_type="application/pdf",
)
def monthly_sales_report(period: str, region_slug: str) -> bytes:
    """Retorna o relatorio PDF de vendas para uma regiao e um periodo."""
    key = (region_slug.title(), period)
    if key not in REPORTS:
        raise ValueError(f"Relatorio nao encontrado para {region_slug=} e {period=}.")
    return REPORTS[key]


if __name__ == "__main__":
    files_server.run()
