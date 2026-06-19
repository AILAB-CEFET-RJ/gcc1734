from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HourlySale:
    product: str
    hour: str
    revenue: float
    region: str


PRODUCT = "X"
REGION = "Sul"
HISTORICAL_HOURLY_AVERAGE = 12_000.0
HISTORICAL_HOURLY_AVERAGE_BY_PRODUCT = {
    PRODUCT: 12_000.0,
    "Y": 9_000.0,
    "Z": 5_600.0,
}

HOURLY_SALES = [
    HourlySale(PRODUCT, "2024-11-18 00:00", 6_120.0, REGION),
    HourlySale(PRODUCT, "2024-11-18 01:00", 6_560.0, REGION),
    HourlySale(PRODUCT, "2024-11-18 02:00", 6_300.0, REGION),
    HourlySale(PRODUCT, "2024-11-18 03:00", 6_780.0, REGION),
    HourlySale(PRODUCT, "2024-11-18 04:00", 7_100.0, REGION),
    HourlySale(PRODUCT, "2024-11-18 05:00", 6_980.0, REGION),
    HourlySale(PRODUCT, "2024-11-18 06:00", 7_050.0, REGION),
    HourlySale(PRODUCT, "2024-11-18 07:00", 6_870.0, REGION),
    HourlySale("Y", "2024-11-18 00:00", 8_900.0, REGION),
    HourlySale("Y", "2024-11-18 01:00", 9_100.0, REGION),
    HourlySale("Z", "2024-11-18 00:00", 5_500.0, REGION),
    HourlySale("Z", "2024-11-18 01:00", 5_620.0, REGION),
]

STOCK_BY_PRODUCT = {
    PRODUCT: {
        "estoque": 0,
        "ultimo_reabastecimento": "3 dias atras",
    },
    "Y": {
        "estoque": 230,
        "ultimo_reabastecimento": "1 dia atras",
    },
    "Z": {
        "estoque": 80,
        "ultimo_reabastecimento": "2 dias atras",
    },
}

INCIDENT_HISTORY = [
    {
        "produto": PRODUCT,
        "data": "2024-06-03",
        "causa": "estoque_zerado",
        "tempo_recuperacao_horas": 5,
    },
    {
        "produto": PRODUCT,
        "data": "2024-09-14",
        "causa": "estoque_zerado",
        "tempo_recuperacao_horas": 3,
    },
]


def format_brl(value: float) -> str:
    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def percent_drop(current_average: float, baseline: float) -> float:
    return (baseline - current_average) / baseline
