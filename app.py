"""
app.py — Smart Inventory Decision Support System
=================================================
Positions the app as a decision support tool, not a prediction oracle.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reorder import compute_reorder_point, dead_capital_cost
from scenarios import run_scenario, compare_scenarios, get_ksa_preset_scenarios, ScenarioConfig

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Inventory — KSA",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Smart Inventory Decision Support")
st.caption(
    "This tool helps you make better stocking decisions under uncertainty. "
    "It does not predict the future — it maps how your risk exposure changes "
    "under different demand and supply scenarios."
)

# ── Sidebar: Data Input ────────────────────────────────────────────────────────

st.sidebar.header("Product Setup")

product_name = st.sidebar.text_input("Product Name", value="Rice (5kg)")
current_stock = st.sidebar.number_input("Current Stock (units)", min_value=0, value=80)
unit_price = st.sidebar.number_input("Unit Price (SAR)", min_value=1.0, value=25.0, step=0.5)
lead_time = st.sidebar.slider("Supplier Lead Time (days)", 1, 14, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("Historical Sales Data")

data_mode = st.sidebar.radio(
    "Data source",
    ["Use sample data", "Enter manually"]
)

if data_mode == "Use sample data":
    # Generate realistic KSA sample data
    np.random.seed(42)
    n_days = 60
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days)
    base_sales = np.random.normal(loc=12, scale=3, size=n_days).clip(min=1)

    # Friday spike
    for i, d in enumerate(dates):
        if d.weekday() == 4:
            base_sales[i] += np.random.randint(5, 10)

    daily_sales = pd.Series(base_sales.round().astype(int), index=dates)
    st.sidebar.success(f"Using 60 days of sample data (avg: {daily_sales.mean():.1f} units/day)")

else:
    raw_input = st.sidebar.text_area(
        "Paste daily sales (one number per line)",
        value="10\n12\n8\n15\n9\n11\n20\n13\n10\n14"
    )
    try:
        values = [float(x.strip()) for x in raw_input.strip().split("\n") if x.strip()]
        daily_sales = pd.Series(values)
        st.sidebar.success(f"{len(daily_sales)} days entered (avg: {daily_sales.mean():.1f} units/day)")
    except Exception:
        st.sidebar.error("Invalid input — enter one number per line")
        st.stop()

# ── Tab Layout ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Reorder Analysis", "🔭 Scenario Simulator", "📈 Sales History"])

# ── TAB 1: Reorder Point Analysis ─────────────────────────────────────────────

with tab1:
    st.subheader("Dynamic Reorder Point")

    col1, col2 = st.columns([1, 2])

    with col1:
        service_level = st.select_slider(
            "Service Level (stockout tolerance)",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}% — {'Low' if x==0.90 else 'Balanced' if x==0.95 else 'Critical'} risk"
        )

        rop_result = compute_reorder_point(daily_sales, lead_time, service_level)

        st.metric("Reorder Point", f"{rop_result['reorder_point']} units")
        st.metric("Safety Stock", f"{rop_result['safety_stock']} units")
        st.metric("Avg Daily Demand", f"{rop_result['avg_daily_demand']} units/day")

        if rop_result['demand_std'] is not None:
            st.metric("Demand Std Dev", f"±{rop_result['demand_std']} units")

        # Status indicator
        if current_stock <= rop_result['reorder_point']:
            st.error(f"🚨 ORDER NOW — Stock ({current_stock}) is at or below reorder point ({rop_result['reorder_point']})")
        elif current_stock <= rop_result['reorder_point'] * 1.3:
            st.warning(f"⚠️ Order soon — stock within 30% of reorder point")
        else:
            st.success(f"✅ Stock level is adequate")

    with col2:
        st.markdown("**What this means:**")
        st.info(rop_result['interpretation'])
        st.caption(f"Demand volatility: {rop_result['demand_volatility']}")

        # Dead capital analysis
        st.markdown("**Dead Capital Check:**")
        dc = dead_capital_cost(
            current_stock,
            rop_result['reorder_point'],
            rop_result['avg_daily_demand'],
            unit_price
        )

        dcol1, dcol2, dcol3 = st.columns(3)
        dcol1.metric("Excess Units", dc['excess_units'])
        dcol2.metric("Days of Excess Supply", dc['days_of_excess_supply'])
        dcol3.metric("Opportunity Cost", f"{dc['dead_capital_SAR']} SAR")

        st.caption(dc['recommendation'])

        st.markdown("---")
        st.markdown(
            "**Methodology note:** Safety stock uses the formula `Z × σ × √(lead time)` "
            "where Z is the service level z-score and σ is the standard deviation of daily demand. "
            "This accounts for demand volatility, not just average demand."
        )

# ── TAB 2: Scenario Simulator ─────────────────────────────────────────────────

with tab2:
    st.subheader("Scenario Simulator — What-If Analysis")
    st.caption(
        "Each scenario recomputes the reorder point and stockout risk "
        "under different demand and supply assumptions."
    )

    mode = st.radio("Mode", ["KSA Preset Scenarios", "Custom Scenario"])

    if mode == "KSA Preset Scenarios":
        presets = get_ksa_preset_scenarios(lead_time)
        comparison_df = compare_scenarios(daily_sales, current_stock, unit_price, presets)

        # Highlight the comparison table
        def highlight_risk(val):
            if isinstance(val, str):
                if "High" in val:
                    return "background-color: #ffe0e0"
                elif "Moderate" in val:
                    return "background-color: #fff3cd"
            if isinstance(val, (int, float)):
                if val > 40:
                    return "color: red; font-weight: bold"
            return ""

        st.dataframe(
            comparison_df.style.applymap(highlight_risk),
            use_container_width=True
        )

        # Visualize reorder points across scenarios
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#2196F3', '#FF9800', '#F44336', '#9C27B0', '#4CAF50', '#795548']
        bars = ax.barh(
            comparison_df["Scenario"],
            comparison_df["Reorder Point"],
            color=colors[:len(comparison_df)]
        )
        ax.axvline(current_stock, color='red', linestyle='--', linewidth=1.5, label=f'Current Stock ({current_stock})')
        ax.set_xlabel("Units")
        ax.set_title("Reorder Point by Scenario vs Current Stock")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Red dashed line = your current stock. "
            "Any bar extending past the red line means you need to order under that scenario."
        )

    else:
        # Custom scenario builder
        st.markdown("**Build your own scenario:**")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demand**")
            demand_mult = st.slider("Demand multiplier", 0.5, 3.0, 1.0, 0.1,
                                    help="1.0 = baseline, 1.5 = 50% higher demand")
            ramadan = st.checkbox("Ramadan active")
            eid = st.checkbox("Eid al-Fitr spike")

        with c2:
            st.markdown("**Supply**")
            delay = st.slider("Extra supply delay (days)", 0, 14, 0)

        with c3:
            st.markdown("**Price**")
            price_chg = st.slider("Price change %", -30, 50, 0)
            elasticity = st.slider("Price elasticity", -2.0, 0.0, -0.5, 0.1,
                                   help="-0.5 = typical grocery staple")

        scenario_name = f"Custom: {demand_mult}x demand, +{delay}d delay, {price_chg}% price"

        custom_config = ScenarioConfig(
            name=scenario_name,
            demand_multiplier=demand_mult,
            ramadan_active=ramadan,
            eid_spike=eid,
            lead_time_days=lead_time,
            supply_delay_days=delay,
            price_change_pct=float(price_chg),
            price_elasticity=elasticity,
        )

        result = run_scenario(daily_sales, current_stock, unit_price, custom_config)

        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Adjusted Demand", f"{result.adjusted_avg_demand} units/day")
        rcol2.metric("Reorder Point", f"{result.reorder_point} units")
        rcol3.metric("Recommended Order", f"{result.recommended_order_qty} units")
        rcol4.metric("Stockout Risk", f"{result.stockout_risk_pct}%")

        if result.notes:
            st.markdown("**Scenario analysis:**")
            for note in result.notes:
                st.write(f"• {note}")

        st.metric("Revenue Impact vs Baseline", f"{result.revenue_impact_pct:+.1f}%")
        st.write(f"Dead Capital Risk: {result.dead_capital_risk}")

# ── TAB 3: Sales History ───────────────────────────────────────────────────────

with tab3:
    st.subheader("Sales History")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(daily_sales.index, daily_sales.values, linewidth=1.5, color='#1976D2')
    ax2.fill_between(daily_sales.index, daily_sales.values, alpha=0.1, color='#1976D2')
    ax2.axhline(daily_sales.mean(), color='orange', linestyle='--', label=f'Mean ({daily_sales.mean():.1f})')
    ax2.set_ylabel("Units Sold")
    ax2.set_title(f"Daily Sales — {product_name}")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Days of Data", len(daily_sales))
    scol2.metric("Avg Daily", f"{daily_sales.mean():.1f}")
    scol3.metric("Peak Day", f"{daily_sales.max():.0f}")
    scol4.metric("Std Dev", f"±{daily_sales.std():.1f}")

    st.dataframe(
        pd.DataFrame({
            "Date": daily_sales.index,
            "Units Sold": daily_sales.values
        }).set_index("Date").tail(14),
        use_container_width=True
    )
