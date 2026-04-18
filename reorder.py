"""
reorder.py — Dynamic Reorder Point Engine
==========================================
Computes statistically-grounded reorder points instead of arbitrary thresholds.

Key distinction from naive approaches:
  - Accounts for demand VOLATILITY (σ), not just average demand
  - Scales safety stock with lead time uncertainty
  - Uses configurable service levels (risk tolerance)

Interview talking point:
  "The reorder point is derived from inventory theory — it's the
   same logic used in supply chain management at scale, adapted
   for data-sparse small retail environments."
"""

import numpy as np
import pandas as pd


# Service level → Z-score mapping
# Z-score = how many standard deviations above mean to buffer
SERVICE_LEVELS = {
    0.90: 1.28,   # 90% — accept some stockouts, minimize holding cost
    0.95: 1.645,  # 95% — balanced (recommended default)
    0.99: 2.326,  # 99% — critical items (e.g., Ramadan staples)
}


def compute_reorder_point(
    daily_sales: pd.Series,
    lead_time_days: int = 3,
    service_level: float = 0.95
) -> dict:
    """
    Compute dynamic reorder point from historical daily sales.

    Parameters
    ----------
    daily_sales   : pd.Series of daily units sold for ONE product
    lead_time_days: days between placing and receiving an order
    service_level : probability of not stocking out (0.90 / 0.95 / 0.99)

    Returns
    -------
    dict with reorder_point, safety_stock, avg_demand, demand_std, interpretation
    """

    if len(daily_sales) < 7:
        # Insufficient data — fall back to simple heuristic
        return _sparse_data_fallback(daily_sales, lead_time_days)

    avg_demand = daily_sales.mean()
    demand_std = daily_sales.std()

    z = SERVICE_LEVELS.get(service_level, 1.645)

    # Core formula
    safety_stock = z * demand_std * np.sqrt(lead_time_days)
    reorder_point = (avg_demand * lead_time_days) + safety_stock

    # Volatility ratio — useful for explaining to shop owners
    cv = demand_std / avg_demand if avg_demand > 0 else 0  # coefficient of variation

    return {
        "reorder_point": round(reorder_point),
        "safety_stock": round(safety_stock),
        "avg_daily_demand": round(avg_demand, 2),
        "demand_std": round(demand_std, 2),
        "lead_time_days": lead_time_days,
        "service_level_pct": int(service_level * 100),
        "demand_volatility": _classify_volatility(cv),
        "interpretation": _generate_interpretation(
            reorder_point, safety_stock, avg_demand, cv, lead_time_days
        )
    }


def _sparse_data_fallback(daily_sales: pd.Series, lead_time_days: int) -> dict:
    """
    When data is too sparse for statistical safety stock,
    fall back to a conservative multiplier and flag it clearly.

    This is intentional design — the app tells the user WHY
    it's using a different method, which builds trust.
    """
    avg_demand = daily_sales.mean() if len(daily_sales) > 0 else 5
    conservative_buffer = avg_demand * 0.5  # 50% buffer under uncertainty
    reorder_point = (avg_demand * lead_time_days) + conservative_buffer

    return {
        "reorder_point": round(reorder_point),
        "safety_stock": round(conservative_buffer),
        "avg_daily_demand": round(avg_demand, 2),
        "demand_std": None,
        "lead_time_days": lead_time_days,
        "service_level_pct": "N/A",
        "demand_volatility": "Unknown (sparse data)",
        "interpretation": (
            f"⚠️ Limited data ({len(daily_sales)} days). "
            f"Using conservative 50% buffer. "
            f"Reorder when stock reaches {round(reorder_point)} units. "
            f"Collect more sales data for statistical accuracy."
        )
    }


def _classify_volatility(cv: float) -> str:
    """Translate coefficient of variation into plain language."""
    if cv < 0.2:
        return "Low — demand is stable and predictable"
    elif cv < 0.5:
        return "Moderate — some day-to-day variation"
    else:
        return "High — demand is erratic, larger safety stock needed"


def _generate_interpretation(rop, safety_stock, avg_demand, cv, lead_time) -> str:
    return (
        f"Reorder when stock drops to {round(rop)} units. "
        f"This covers {lead_time} days of avg demand ({round(avg_demand)}/day) "
        f"plus a safety buffer of {round(safety_stock)} units. "
        f"Demand volatility: {_classify_volatility(cv)}."
    )


# ── Dead Capital Calculator ────────────────────────────────────────────────────

def dead_capital_cost(
    current_stock: int,
    reorder_point: int,
    avg_daily_demand: float,
    unit_price: float
) -> dict:
    """
    Estimate the cost of holding EXCESS inventory above the optimal level.

    Dead capital = money tied up in stock that isn't generating revenue.
    This is a real pain point for small KSA retailers.
    """
    # Days of excess supply
    excess_units = max(0, current_stock - reorder_point)
    days_of_excess = excess_units / avg_daily_demand if avg_daily_demand > 0 else 0

    # Opportunity cost (assume 8% annual — rough cost of capital for SMEs)
    annual_rate = 0.08
    daily_rate = annual_rate / 365
    opportunity_cost = excess_units * unit_price * daily_rate * days_of_excess

    return {
        "excess_units": round(excess_units),
        "days_of_excess_supply": round(days_of_excess, 1),
        "dead_capital_SAR": round(opportunity_cost, 2),
        "recommendation": (
            "Stock level is optimal." if excess_units == 0
            else f"Consider reducing next order by ~{round(excess_units)} units "
                 f"to free up {round(excess_units * unit_price)} SAR in working capital."
        )
    }
