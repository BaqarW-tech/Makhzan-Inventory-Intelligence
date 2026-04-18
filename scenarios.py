"""
scenarios.py — Scenario Simulation Engine
==========================================
Answers counterfactual questions like:
  "What if Ramadan demand rises 30%?"
  "What if my supplier delays by 5 days?"
  "What if I raise prices by 15%?"

This is the economics layer. Forecasting tells you what probably
happens. Scenario simulation tells you what you should DO given
different possible futures.

Interview talking point:
  "The simulation uses sensitivity analysis — the same technique
   used in financial modelling and policy analysis — to map how
   outcomes change under different assumptions."
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioConfig:
    """
    All the levers the user can pull.
    Defaults represent the baseline (no change).
    """
    name: str = "Baseline"

    # Demand shocks
    demand_multiplier: float = 1.0        # 1.3 = +30% demand
    ramadan_active: bool = False          # toggles Ramadan uplift
    ramadan_demand_boost: float = 1.4    # Ramadan typically +40% on staples
    eid_spike: bool = False              # one-week Eid spike
    eid_demand_boost: float = 1.8        # Eid is sharper / shorter than Ramadan

    # Supply shocks
    lead_time_days: int = 3              # baseline supplier lead time
    supply_delay_days: int = 0           # additional delay (port, customs, etc.)

    # Price sensitivity
    price_change_pct: float = 0.0        # +15 = 15% price increase
    price_elasticity: float = -0.5       # how much demand drops per 1% price rise
                                         # -0.5 is typical for grocery staples in KSA

    # Service level (risk tolerance)
    service_level: float = 0.95


@dataclass
class ScenarioResult:
    """Output of one scenario run."""
    scenario_name: str
    adjusted_avg_demand: float
    adjusted_lead_time: int
    reorder_point: int
    safety_stock: int
    recommended_order_qty: int
    stockout_risk_pct: float
    dead_capital_risk: str
    revenue_impact_pct: float
    notes: list = field(default_factory=list)


def run_scenario(
    baseline_daily_sales: pd.Series,
    current_stock: int,
    unit_price: float,
    config: ScenarioConfig
) -> ScenarioResult:
    """
    Apply a scenario config to baseline sales data and return
    adjusted inventory recommendations.

    Parameters
    ----------
    baseline_daily_sales : pd.Series — historical daily units sold
    current_stock        : int — units currently on hand
    unit_price           : float — selling price per unit (SAR)
    config               : ScenarioConfig — the scenario to simulate
    """

    notes = []

    # ── Step 1: Adjust demand ──────────────────────────────────────────────────

    avg_demand = baseline_daily_sales.mean()
    demand_std = baseline_daily_sales.std() if len(baseline_daily_sales) >= 7 else avg_demand * 0.3

    # Apply general demand multiplier
    adjusted_demand = avg_demand * config.demand_multiplier

    # Apply Ramadan uplift
    if config.ramadan_active:
        adjusted_demand *= config.ramadan_demand_boost
        notes.append(
            f"Ramadan active: demand boosted {int((config.ramadan_demand_boost - 1) * 100)}% "
            f"→ {round(adjusted_demand, 1)} units/day"
        )

    # Apply Eid spike (overrides Ramadan if both set — Eid is the peak)
    if config.eid_spike:
        adjusted_demand = avg_demand * config.eid_demand_boost
        notes.append(
            f"Eid spike: demand surges {int((config.eid_demand_boost - 1) * 100)}% "
            f"→ {round(adjusted_demand, 1)} units/day (short window, order now)"
        )

    # Apply price sensitivity
    if config.price_change_pct != 0:
        demand_reduction_pct = config.price_change_pct * abs(config.price_elasticity)
        price_adjustment = 1 - (demand_reduction_pct / 100)
        adjusted_demand *= price_adjustment
        direction = "reduces" if config.price_change_pct > 0 else "increases"
        notes.append(
            f"Price {'+' if config.price_change_pct > 0 else ''}{config.price_change_pct}% "
            f"{direction} demand by ~{round(demand_reduction_pct, 1)}% "
            f"(elasticity = {config.price_elasticity})"
        )

    # ── Step 2: Adjust lead time ───────────────────────────────────────────────

    adjusted_lead_time = config.lead_time_days + config.supply_delay_days

    if config.supply_delay_days > 0:
        notes.append(
            f"Supply delay of {config.supply_delay_days} days increases "
            f"stockout exposure window to {adjusted_lead_time} days"
        )

    # ── Step 3: Recompute ROP under this scenario ──────────────────────────────

    from reorder import SERVICE_LEVELS

    z = SERVICE_LEVELS.get(config.service_level, 1.645)

    # Demand std scales with adjusted demand (proportional assumption)
    adjusted_std = demand_std * (adjusted_demand / avg_demand) if avg_demand > 0 else demand_std

    safety_stock = z * adjusted_std * np.sqrt(adjusted_lead_time)
    reorder_point = (adjusted_demand * adjusted_lead_time) + safety_stock

    # ── Step 4: Recommended order quantity (Economic reasoning) ───────────────

    # How many days until next reorder window? Assume 7-day review cycle.
    review_period = 7
    recommended_order = (adjusted_demand * (adjusted_lead_time + review_period)) + safety_stock

    # ── Step 5: Stockout risk ──────────────────────────────────────────────────

    # Days of supply remaining at adjusted demand rate
    days_of_supply = current_stock / adjusted_demand if adjusted_demand > 0 else 999

    # Probability of stocking out before reorder arrives
    if days_of_supply >= adjusted_lead_time:
        stockout_risk_pct = round((1 - config.service_level) * 100, 1)
    else:
        # Already below ROP — risk is high and increasing
        deficit = adjusted_lead_time - days_of_supply
        stockout_risk_pct = min(95, round(30 + (deficit * 15), 1))
        notes.append(
            f"⚠️ Current stock ({current_stock} units) is below ROP ({round(reorder_point)}) "
            f"— order immediately."
        )

    # ── Step 6: Revenue impact vs baseline ────────────────────────────────────

    baseline_revenue = avg_demand * unit_price
    adjusted_price = unit_price * (1 + config.price_change_pct / 100)
    scenario_revenue = adjusted_demand * adjusted_price
    revenue_impact_pct = round(((scenario_revenue - baseline_revenue) / baseline_revenue) * 100, 1)

    # ── Step 7: Dead capital risk flag ────────────────────────────────────────

    if current_stock > reorder_point * 2:
        dead_capital_risk = "High — significant over-stock relative to scenario demand"
    elif current_stock > reorder_point * 1.2:
        dead_capital_risk = "Moderate — some excess inventory"
    else:
        dead_capital_risk = "Low — stock aligns with projected demand"

    return ScenarioResult(
        scenario_name=config.name,
        adjusted_avg_demand=round(adjusted_demand, 2),
        adjusted_lead_time=adjusted_lead_time,
        reorder_point=round(reorder_point),
        safety_stock=round(safety_stock),
        recommended_order_qty=round(recommended_order),
        stockout_risk_pct=stockout_risk_pct,
        dead_capital_risk=dead_capital_risk,
        revenue_impact_pct=revenue_impact_pct,
        notes=notes
    )


def compare_scenarios(
    baseline_daily_sales: pd.Series,
    current_stock: int,
    unit_price: float,
    scenarios: list[ScenarioConfig]
) -> pd.DataFrame:
    """
    Run multiple scenarios and return a comparison table.
    This is what gets displayed as the side-by-side dashboard view.
    """
    results = []
    for config in scenarios:
        r = run_scenario(baseline_daily_sales, current_stock, unit_price, config)
        results.append({
            "Scenario": r.scenario_name,
            "Avg Demand (units/day)": r.adjusted_avg_demand,
            "Reorder Point": r.reorder_point,
            "Safety Stock": r.safety_stock,
            "Order Qty": r.recommended_order_qty,
            "Stockout Risk %": r.stockout_risk_pct,
            "Revenue Impact %": r.revenue_impact_pct,
            "Dead Capital Risk": r.dead_capital_risk,
        })

    return pd.DataFrame(results)


# ── Preset Scenario Library (KSA-specific) ────────────────────────────────────

def get_ksa_preset_scenarios(lead_time: int = 3) -> list[ScenarioConfig]:
    """
    Ready-made scenarios for KSA retail context.
    The user can select these from a dropdown rather than configuring manually.
    """
    return [
        ScenarioConfig(
            name="Baseline (Normal Week)",
            lead_time_days=lead_time
        ),
        ScenarioConfig(
            name="Ramadan — Staple Goods",
            ramadan_active=True,
            ramadan_demand_boost=1.4,
            lead_time_days=lead_time,
            service_level=0.99  # higher service level during Ramadan
        ),
        ScenarioConfig(
            name="Eid al-Fitr Peak",
            eid_spike=True,
            eid_demand_boost=1.8,
            lead_time_days=lead_time,
            service_level=0.99
        ),
        ScenarioConfig(
            name="Supply Disruption (Port Delay)",
            supply_delay_days=4,
            lead_time_days=lead_time
        ),
        ScenarioConfig(
            name="Price Increase +15%",
            price_change_pct=15.0,
            price_elasticity=-0.5,
            lead_time_days=lead_time
        ),
        ScenarioConfig(
            name="Ramadan + Supply Delay (Stress Test)",
            ramadan_active=True,
            ramadan_demand_boost=1.4,
            supply_delay_days=3,
            lead_time_days=lead_time,
            service_level=0.99
        ),
    ]
