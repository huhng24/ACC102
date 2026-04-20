"""
Rate Shock Explorer
A Streamlit app to explore the impact of Fed rate hikes on global markets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rate Shock Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 5px 0;
    }
    .hike-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2196F3;
        border-radius: 5px;
        padding: 10px 15px;
        margin: 8px 0;
        font-size: 0.9em;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = ""

ASSETS = ["SPY", "GLD", "CNY", "DXY", "DGS10"]
ASSET_LABELS = {
    "SPY":   "US Stocks (SPY)",
    "GLD":   "Gold (GLD)",
    "CNY":   "CNY/USD",
    "DXY":   "USD Index (DXY)",
    "DGS10": "10Y Treasury Yield",
}

CYCLE_COLORS = {
    "2015-2018": "steelblue",
    "2022-2023": "tomato",
}

# Background context for each hike
HIKE_CONTEXT = {
    "2015-12-16": "First rate hike in nearly a decade. Economy recovering from GFC. Unemployment at 5%.",
    "2016-12-14": "Only hike of 2016. Trump election victory, reflation trade in full swing.",
    "2017-03-15": "Fed signals faster tightening. Inflation gradually approaching 2% target.",
    "2017-06-14": "Steady tightening cycle. US economy growing at moderate pace.",
    "2017-12-13": "Tax reform boost to growth expectations. Third hike of the year.",
    "2018-03-21": "New Fed Chair Powell's first hike. Trade war fears beginning.",
    "2018-06-13": "Tariffs escalating. Fed confident in economy despite external risks.",
    "2018-09-26": "Economy running hot. Unemployment at 18-year low of 3.7%.",
    "2018-12-19": "Final hike of cycle amid market turmoil. S&P 500 fell ~20% in Q4 2018.",
    "2022-03-16": "First hike since 2018. CPI at 7.9%, highest since 1982. Russia-Ukraine war ongoing.",
    "2022-05-04": "Largest hike since 2000 (50bp). Fed acknowledges inflation is not transitory.",
    "2022-06-15": "First 75bp hike since 1994. CPI hit 9.1% — 40-year high. Shock to markets.",
    "2022-07-27": "Second consecutive 75bp hike. Fed prioritises inflation over growth.",
    "2022-09-21": "Third 75bp hike. Dollar surging. Emerging markets under severe pressure.",
    "2022-11-02": "Fourth 75bp hike. Signs of cooling inflation beginning to emerge.",
    "2022-12-14": "Step down to 50bp. Fed signals peak rate approaching.",
    "2023-02-01": "Back to 25bp. Disinflation trend confirmed. Soft landing hopes rising.",
    "2023-03-22": "Silicon Valley Bank collapse days earlier. Fed hikes despite banking stress.",
    "2023-05-03": "Possibly the last hike. Banking sector stabilised. CPI falling steadily.",
    "2023-07-26": "Final hike of the cycle. Fed funds rate peaks at 5.25–5.50%.",
}

# Market expectations (was the hike a surprise?)
HIKE_SURPRISE = {
    "2015-12-16": "In line with expectations",
    "2016-12-14": "In line with expectations",
    "2017-03-15": "Slightly hawkish surprise",
    "2017-06-14": "In line with expectations",
    "2017-12-13": "In line with expectations",
    "2018-03-21": "In line with expectations",
    "2018-06-13": "In line with expectations",
    "2018-09-26": "In line with expectations",
    "2018-12-19": "Hawkish surprise — market expected a pause",
    "2022-03-16": "In line with expectations",
    "2022-05-04": "In line with expectations",
    "2022-06-15": "Hawkish surprise — market expected 50bp",
    "2022-07-27": "In line with expectations",
    "2022-09-21": "In line with expectations",
    "2022-11-02": "In line with expectations",
    "2022-12-14": "Dovish surprise — some expected 75bp",
    "2023-02-01": "In line with expectations",
    "2023-03-22": "Hawkish surprise — SVB crisis led many to expect a pause",
    "2023-05-03": "In line with expectations",
    "2023-07-26": "In line with expectations",
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices_clean.csv"), parse_dates=["date"])
    hikes  = pd.read_csv(os.path.join(DATA_DIR, "fed_hike_events.csv"), parse_dates=["date"])

    def categorise_bp(bp):
        if bp == 25:
            return "Small (25bp)"
        elif bp == 50:
            return "Medium (50bp)"
        else:
            return "Large (75bp)"

    hikes["size_cat"] = hikes["rate_change_bp"].apply(categorise_bp)
    return prices, hikes

prices, hikes = load_data()

# ── Core analysis function ────────────────────────────────────────────────────
@st.cache_data
def get_event_windows(asset, pre=5, post=30):
    results = {}
    price_series = prices.set_index("date")[asset].dropna()
    all_dates = price_series.index.tolist()

    for _, row in hikes.iterrows():
        hike_date = row["date"]
        future_dates = [d for d in all_dates if d >= hike_date]
        if not future_dates:
            continue
        nearest = future_dates[0]
        idx = all_dates.index(nearest)

        start_idx = idx - pre
        end_idx   = idx + post + 1
        if start_idx < 0 or end_idx > len(all_dates):
            continue

        window_dates  = all_dates[start_idx:end_idx]
        window_prices = price_series.loc[window_dates]
        base_price    = price_series.loc[nearest]

        cum_returns = ((window_prices / base_price) - 1) * 100
        cum_returns.index = range(-pre, len(window_dates) - pre)

        label = hike_date.strftime("%Y-%m-%d") + f" (+{row['rate_change_bp']}bp)"
        results[label] = cum_returns

    return pd.DataFrame(results)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    # Reset button
    if st.button("🔄 Reset to Default"):
        st.session_state["selected_assets"] = ["SPY", "GLD", "DXY"]
        st.session_state["selected_cycle"]  = "All"
        st.session_state["pre_days"]        = 5
        st.session_state["post_days"]       = 30
        st.session_state["horizon"]         = 10
        st.rerun()

    st.subheader("Asset Selection")
    selected_assets = st.multiselect(
        "Select assets to display:",
        options=ASSETS,
        default=st.session_state.get("selected_assets", ["SPY", "GLD", "DXY"]),
        format_func=lambda x: ASSET_LABELS[x],
        key="selected_assets"
    )

    st.subheader("Cycle Filter")
    selected_cycle = st.radio(
        "Rate hike cycle:",
        options=["All", "2015-2018", "2022-2023"],
        index=0,
        key="selected_cycle"
    )

    st.subheader("Window Period")
    pre_days = st.slider(
        "Days before hike:",
        min_value=1, max_value=20, value=st.session_state.get("pre_days", 5),
        key="pre_days"
    )
    post_days = st.slider(
        "Days after hike:",
        min_value=5, max_value=60, value=st.session_state.get("post_days", 30),
        key="post_days"
    )

    st.subheader("Analysis Horizon")
    horizon = st.select_slider(
        "Observation point (days after hike):",
        options=[1, 5, 10, 20, 30],
        value=st.session_state.get("horizon", 10),
        key="horizon"
    )

    st.markdown("---")
    st.caption("Data sources: Stooq, FRED, Federal Reserve FOMC records")

# Filter hikes by cycle
if selected_cycle != "All":
    hikes_filtered = hikes[hikes["cycle"] == selected_cycle]
else:
    hikes_filtered = hikes

# ── Main content ──────────────────────────────────────────────────────────────
st.title("📈 Rate Shock Explorer")
st.markdown("*How Federal Reserve rate hikes ripple through global markets*")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Event Windows",
    "📊 Average Impact",
    "🔎 Hike Size vs Reaction",
    "🗓️ Single Hike Detail",
    "🗃️ Data Table",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Event Window
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class='info-box'>
    📌 <b>What this shows:</b> Each line is one Fed rate hike. The x-axis shows trading days 
    relative to the hike date (Day 0). The y-axis shows cumulative return from the hike day. 
    The dashed black line is the average across all hikes. <b>Blue = 2015–2018 cycle, Red = 2022–2023 cycle.</b>
    </div>
    """, unsafe_allow_html=True)

    if not selected_assets:
        st.warning("Please select at least one asset from the sidebar.")
    else:
        show_avg = st.toggle("Show average line", value=True)

        for asset in selected_assets:
            df = get_event_windows(asset, pre_days, post_days)

            # Filter by cycle
            if selected_cycle != "All":
                cols_to_keep = [
                    col for col in df.columns
                    if hikes[hikes["date"] == pd.Timestamp(col.split(" ")[0])]["cycle"].values[0] == selected_cycle
                ]
                df = df[cols_to_keep]

            fig = go.Figure()

            for col in df.columns:
                date_str = col.split(" ")[0]
                hike_row = hikes[hikes["date"] == pd.Timestamp(date_str)]
                if hike_row.empty:
                    continue
                cycle = hike_row.iloc[0]["cycle"]
                color = CYCLE_COLORS.get(cycle, "gray")

                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    mode="lines", name=col,
                    line=dict(color=color, width=1.2),
                    opacity=0.55,
                    hovertemplate=f"<b>{col}</b><br>Day: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>"
                ))

            if show_avg and not df.empty:
                avg = df.mean(axis=1)
                fig.add_trace(go.Scatter(
                    x=avg.index, y=avg,
                    mode="lines", name="Average",
                    line=dict(color="black", width=2.5, dash="dash"),
                    hovertemplate="<b>Average</b><br>Day: %{x}<br>Return: %{y:.2f}%<extra></extra>"
                ))

            fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
            fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1,
                          annotation_text="Hike Day", annotation_position="top right")

            fig.update_layout(
                title=f"{ASSET_LABELS[asset]} — Cumulative Return Around Each Fed Hike",
                xaxis_title="Trading Days Relative to Hike",
                yaxis_title="Cumulative Return (%)",
                legend_title="Hike Event",
                hovermode="x unified",
                template="plotly_white",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Average Impact
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class='info-box'>
    📌 <b>What this shows:</b> The average cumulative return across all hikes, 
    for each asset at different time horizons after the hike. Use the sidebar to 
    filter by cycle or change the observation window.
    </div>
    """, unsafe_allow_html=True)

    group_by = st.radio("Group bars by:", ["Horizon", "Cycle"], horizontal=True)

    days_list = [1, 5, 10, 20, 30]
    rows = []

    for asset in (selected_assets if selected_assets else ASSETS):
        df_full = get_event_windows(asset, pre_days, max(days_list))
        for d in days_list:
            if d not in df_full.index:
                continue
            for col in df_full.columns:
                date_str = col.split(" ")[0]
                hike_row = hikes_filtered[hikes_filtered["date"] == pd.Timestamp(date_str)]
                if hike_row.empty:
                    continue
                rows.append({
                    "Asset": ASSET_LABELS[asset],
                    "Day": f"Day +{d}",
                    "Cycle": hike_row.iloc[0]["cycle"],
                    "Return": df_full.loc[d, col],
                })

    if rows:
        summary_df = pd.DataFrame(rows)
        avg_df = summary_df.groupby(["Asset", "Day", "Cycle"])["Return"].mean().reset_index()

        if group_by == "Horizon":
            plot_df = avg_df.groupby(["Asset", "Day"])["Return"].mean().reset_index()
            fig2 = px.bar(
                plot_df, x="Asset", y="Return", color="Day",
                barmode="group",
                title="Average Cumulative Return After Fed Hike — By Asset and Horizon",
                labels={"Return": "Avg Return (%)"},
                template="plotly_white",
                height=480,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
        else:
            plot_df = avg_df.groupby(["Asset", "Cycle"])["Return"].mean().reset_index()
            fig2 = px.bar(
                plot_df, x="Asset", y="Return", color="Cycle",
                barmode="group",
                title="Average Cumulative Return After Fed Hike — By Asset and Cycle",
                labels={"Return": "Avg Return (%)"},
                template="plotly_white",
                height=480,
                color_discrete_map=CYCLE_COLORS,
            )

        fig2.add_hline(y=0, line_dash="dot", line_color="gray")
        fig2.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)

        # Summary table
        with st.expander("📋 View summary table"):
            pivot = summary_df.groupby(["Asset", "Day"])["Return"].mean().unstack().round(2)
            st.dataframe(pivot, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Hike Size vs Reaction
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class='info-box'>
    📌 <b>What this shows:</b> Average asset return at the selected horizon, 
    broken down by hike size (25bp / 50bp / 75bp). Larger hikes don't always 
    produce larger market reactions — explore how each asset responds differently.
    </div>
    """, unsafe_allow_html=True)

    rows3 = []
    for asset in (selected_assets if selected_assets else ASSETS):
        df3 = get_event_windows(asset, pre_days, horizon)
        if horizon not in df3.index:
            continue
        for col in df3.columns:
            date_str = col.split(" ")[0]
            hike_row = hikes_filtered[hikes_filtered["date"] == pd.Timestamp(date_str)]
            if hike_row.empty:
                continue
            rows3.append({
                "asset": ASSET_LABELS[asset],
                "return": df3.loc[horizon, col],
                "size_cat": hike_row.iloc[0]["size_cat"],
                "bp": hike_row.iloc[0]["rate_change_bp"],
            })

    if rows3:
        df3_all = pd.DataFrame(rows3)
        avg3 = df3_all.groupby(["asset", "size_cat", "bp"])["return"].mean().reset_index()
        avg3 = avg3.sort_values("bp")

        size_cats = ["Small (25bp)", "Medium (50bp)", "Large (75bp)"]
        colors3   = {"Small (25bp)": "steelblue", "Medium (50bp)": "orange", "Large (75bp)": "tomato"}

        fig3 = go.Figure()
        for cat in size_cats:
            subset = avg3[avg3["size_cat"] == cat]
            if subset.empty:
                continue
            fig3.add_trace(go.Bar(
                y=subset["asset"], x=subset["return"],
                name=cat, orientation="h",
                marker_color=colors3[cat],
                text=subset["return"].round(2).astype(str) + "%",
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Avg Return: %{x:.2f}%<extra>" + cat + "</extra>",
            ))

        fig3.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
        fig3.update_layout(
            title=f"Average Asset Return at Day +{horizon} — By Hike Size",
            xaxis_title="Average Cumulative Return (%)",
            barmode="group",
            template="plotly_white",
            height=480,
            legend_title="Hike Size",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=160),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Single Hike Detail
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class='info-box'>
    📌 <b>What this shows:</b> Deep-dive into a single Fed hike. 
    Select any hike to see how all assets responded, plus background context 
    and market expectations at the time.
    </div>
    """, unsafe_allow_html=True)

    hike_options = {
        row["date"].strftime("%Y-%m-%d") + f" | +{row['rate_change_bp']}bp → {row['rate_after_pct']}%": row["date"].strftime("%Y-%m-%d")
        for _, row in hikes.iterrows()
    }

    selected_label = st.selectbox("Select a Fed hike event:", options=list(hike_options.keys()))
    selected_date  = hike_options[selected_label]
    selected_hike  = hikes[hikes["date"] == pd.Timestamp(selected_date)].iloc[0]

    # Info cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Date", selected_date)
    with col2:
        st.metric("Hike Size", f"+{selected_hike['rate_change_bp']}bp")
    with col3:
        st.metric("Rate After", f"{selected_hike['rate_after_pct']}%")
    with col4:
        st.metric("Cycle", selected_hike["cycle"])

    # Context card
    context = HIKE_CONTEXT.get(selected_date, "No context available.")
    surprise = HIKE_SURPRISE.get(selected_date, "Unknown")
    st.markdown(f"""
    <div class='hike-card'>
    <b>📰 Market Context:</b> {context}<br><br>
    <b>🎯 Market Expectation:</b> {surprise}
    </div>
    """, unsafe_allow_html=True)

    # Multi-asset response chart
    fig4 = go.Figure()
    assets_to_show = selected_assets if selected_assets else ASSETS

    for asset in assets_to_show:
        df4 = get_event_windows(asset, pre_days, post_days)
        matching_cols = [col for col in df4.columns if col.startswith(selected_date)]
        if not matching_cols:
            continue
        series = df4[matching_cols[0]]
        fig4.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines+markers",
            name=ASSET_LABELS[asset],
            hovertemplate=f"<b>{ASSET_LABELS[asset]}</b><br>Day: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>",
            line=dict(width=2),
            marker=dict(size=4),
        ))

    fig4.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig4.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1,
                   annotation_text="Hike Day", annotation_position="top right")

    fig4.update_layout(
        title=f"All Asset Responses — {selected_date} (+{selected_hike['rate_change_bp']}bp)",
        xaxis_title="Trading Days Relative to Hike",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
        template="plotly_white",
        height=480,
        legend_title="Asset",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Data Table
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
    <div class='info-box'>
    📌 <b>What this shows:</b> The cleaned price data used in this analysis. 
    Filter by date range and download a CSV of the selected data.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
    with col_b:
        end_date = st.date_input("End date", value=pd.Timestamp("2024-12-31"))

    filtered = prices[
        (prices["date"] >= pd.Timestamp(start_date)) &
        (prices["date"] <= pd.Timestamp(end_date))
    ].copy()
    filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")

    st.dataframe(filtered, use_container_width=True, height=400)
    st.caption(f"Showing {len(filtered):,} rows")

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv,
        file_name=f"rate_shock_data_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Fed Hike Events Reference")
    hikes_display = hikes.copy()
    hikes_display["date"] = hikes_display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(hikes_display, use_container_width=True)
