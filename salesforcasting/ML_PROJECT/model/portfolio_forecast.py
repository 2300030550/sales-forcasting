import pickle
import pandas as pd
import os
from utils.feature_pipeline import build_features

# =====================================
# LOAD MODELS
# =====================================
# =====================================
# LOAD MODELS (Correct Path)
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model")

model_p50 = pickle.load(open(os.path.join(model_path, "model_p50.pkl"), "rb"))
model_p10 = pickle.load(open(os.path.join(model_path, "model_p10.pkl"), "rb"))
model_p90 = pickle.load(open(os.path.join(model_path, "model_p90.pkl"), "rb"))

feature_columns = pickle.load(open(os.path.join(model_path, "feature_columns.pkl"), "rb"))

# =====================================
# LOAD DATA
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "full_data.csv")

df = pd.read_csv(data_path)
df = build_features(df)


# =====================================
# PORTFOLIO FORECAST FUNCTION
# =====================================

def forecast_portfolio(quarter):

    quarter_df = df[df["Quarter"] == quarter].copy()

    if quarter_df.empty:
        return "Invalid quarter"

    X = quarter_df[feature_columns]

    # =====================================
    # QUANTILE FORECASTS
    # =====================================

    quarter_df["P50_Units"] = model_p50.predict(X)
    quarter_df["P10_Units"] = model_p10.predict(X)
    quarter_df["P90_Units"] = model_p90.predict(X)

    # Primary forecast
    quarter_df["ForecastedUnits"] = quarter_df["P50_Units"]

    # =====================================
    # UNCERTAINTY METRICS
    # =====================================

    quarter_df["UncertaintyRange"] = (
        quarter_df["P90_Units"] - quarter_df["P10_Units"]
    )

    quarter_df["UncertaintyPercent"] = (
        quarter_df["UncertaintyRange"] /
        (quarter_df["P50_Units"] + 1)
    ) * 100

    # Volatility classification
    quarter_df["DemandVolatility"] = "Stable"

    quarter_df.loc[
        quarter_df["UncertaintyPercent"] > 25,
        "DemandVolatility"
    ] = "Highly Volatile"

    quarter_df.loc[
        quarter_df["UncertaintyPercent"] < 10,
        "DemandVolatility"
    ] = "Highly Predictable"

    # =====================================
    # REVENUE CALCULATION
    # =====================================

    quarter_df["ForecastedRevenue"] = (
        quarter_df["ForecastedUnits"] * quarter_df["CurrentPrice"]
    )

    # =====================================
    # COST & PROFIT MODELING
    # =====================================

    quarter_df["UnitCost"] = quarter_df["LaunchPrice"] * 0.65

    quarter_df["ForecastedProfit"] = (
        (quarter_df["CurrentPrice"] - quarter_df["UnitCost"]) *
        quarter_df["ForecastedUnits"]
    )

    quarter_df["MarginPercent"] = (
        (quarter_df["CurrentPrice"] - quarter_df["UnitCost"]) /
        quarter_df["CurrentPrice"]
    ) * 100

    # =====================================
    # GROWTH VS LAST QUARTER
    # =====================================

    prev_df = df[df["Quarter"] == quarter - 1]

    if not prev_df.empty:
        prev_sales = prev_df[["ProductID", "UnitsSold"]]

        quarter_df = quarter_df.merge(
            prev_sales,
            on="ProductID",
            how="left",
            suffixes=("", "_Prev")
        )

        quarter_df["GrowthVsLastQuarter"] = (
            (quarter_df["ForecastedUnits"] -
             quarter_df["UnitsSold_Prev"]) /
            (quarter_df["UnitsSold_Prev"] + 1)
        ) * 100
    else:
        quarter_df["GrowthVsLastQuarter"] = 0

    # =====================================
    # RISK CLASSIFICATION
    # =====================================

    quarter_df["RiskCategory"] = "Medium"

    quarter_df.loc[
        quarter_df["GrowthVsLastQuarter"] < -10,
        "RiskCategory"
    ] = "High Risk"

    quarter_df.loc[
        quarter_df["GrowthVsLastQuarter"] > 15,
        "RiskCategory"
    ] = "Growth Opportunity"

    # =====================================
    # REVENUE RANKING
    # =====================================

    quarter_df["RevenueRank"] = (
        quarter_df["ForecastedRevenue"]
        .rank(ascending=False)
    )

    # =====================================
    # FINAL OUTPUT
    # =====================================

    return quarter_df.sort_values("RevenueRank")[
        [
            "ProductID",
            "P10_Units",
            "P50_Units",
            "P90_Units",
            "UncertaintyRange",
            "UncertaintyPercent",
            "DemandVolatility",
            "ForecastedRevenue",
            "ForecastedProfit",
            "MarginPercent",
            "GrowthVsLastQuarter",
            "RiskCategory",
            "RevenueRank"
        ]
    ]