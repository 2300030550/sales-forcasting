import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
from model.portfolio_forecast import forecast_portfolio
st.set_page_config(layout="wide")
st.title("📊 Enterprise Sales Intelligence Dashboard")

# =====================================
# QUARTER SELECTION
# =====================================

quarter = st.sidebar.slider("Select Quarter", 1, 12, 10)

portfolio_df = forecast_portfolio(quarter)

# =====================================
# KPI SECTION
# =====================================

total_revenue = portfolio_df["ForecastedRevenue"].sum()
total_profit = portfolio_df["ForecastedProfit"].sum()
avg_margin = portfolio_df["MarginPercent"].mean()
high_risk = (portfolio_df["RiskCategory"] == "High Risk").sum()
growth_skus = (portfolio_df["RiskCategory"] == "Growth Opportunity").sum()
volatile_skus = (portfolio_df["DemandVolatility"] == "Highly Volatile").sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Revenue", f"{total_revenue:,.0f}")
col2.metric("Total Profit", f"{total_profit:,.0f}")
col3.metric("Avg Margin %", f"{avg_margin:.2f}")
col4.metric("High Risk SKUs", high_risk)
col5.metric("Growth SKUs", growth_skus)

st.divider()

# =====================================
# RISK DISTRIBUTION
# =====================================

st.subheader("Risk Distribution")

risk_fig = px.pie(
    portfolio_df,
    names="RiskCategory",
    title="Risk Category Distribution"
)

st.plotly_chart(risk_fig, use_container_width=True)

# =====================================
# VOLATILITY DISTRIBUTION
# =====================================

st.subheader("Demand Volatility Distribution")

vol_fig = px.pie(
    portfolio_df,
    names="DemandVolatility",
    title="Demand Volatility"
)

st.plotly_chart(vol_fig, use_container_width=True)

# =====================================
# TOP PRODUCTS
# =====================================

st.subheader("Top 10 Revenue Products")

top10 = portfolio_df.sort_values(
    "ForecastedRevenue",
    ascending=False
).head(10)

st.dataframe(top10)

# =====================================
# HEATMAP STYLE SCATTER
# =====================================

st.subheader("Revenue vs Margin")

scatter_fig = px.scatter(
    portfolio_df,
    x="MarginPercent",
    y="ForecastedRevenue",
    color="RiskCategory",
    size="UncertaintyPercent",
    hover_data=["ProductID"],
    title="Revenue vs Margin vs Risk"
)

st.plotly_chart(scatter_fig, use_container_width=True)