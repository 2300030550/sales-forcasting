import pandas as pd
import numpy as np


def build_features(df):

    df = df.copy()

    # ==============================
    # 1️⃣ PRODUCT PERFORMANCE INDEX
    # ==============================

    df["PerformanceIndex"] = (
        0.4 * df["RAM"] +
        0.3 * (df["Storage"] / 64) +
        0.3 * df["ProcessorScore"]
    )

    df["CameraScore"] = (
        0.7 * (df["CameraMP"] / 10) +
        0.3 * df["PremiumMaterial"]
    )

    # ==============================
    # 2️⃣ PRICING FEATURES
    # ==============================

    df["DiscountPercent"] = (
        (df["LaunchPrice"] - df["CurrentPrice"]) / df["LaunchPrice"]
    )

    df["MarketingIntensity"] = (
        df["MarketingSpend"] / (df["UnitsSold"] + 1)
    )

    # ==============================
    # 3️⃣ CUSTOMER BUZZ SCORE
    # ==============================

    df["CustomerBuzzScore"] = (
        0.4 * df["AvgRating"] +
        0.3 * np.log(df["ReviewVolume"] + 1) +
        0.3 * df["SentimentScore"]
    )

    # ==============================
    # 4️⃣ LIFECYCLE MODELING
    # ==============================

    df["ProductAge"] = df["Quarter"] - df["LaunchQuarter"]

    df["LifecycleStage"] = np.where(
        df["ProductAge"] <= 2, 0,   # Early
        np.where(df["ProductAge"] <= 6, 1, 2)  # Growth / Maturity
    )

    # ==============================
    # 5️⃣ TIME-SERIES LAG FEATURES
    # ==============================

    df = df.sort_values(["ProductID", "Quarter"])

    df["Lag_1"] = df.groupby("ProductID")["UnitsSold"].shift(1)
    df["Lag_2"] = df.groupby("ProductID")["UnitsSold"].shift(2)

    df["RollingMean_3"] = (
        df.groupby("ProductID")["UnitsSold"]
        .rolling(3)
        .mean()
        .reset_index(0, drop=True)
    )

    # Fill early missing values
    df = df.bfill()

    # ==============================
    # 6️⃣ MARKET ADJUSTED SCORE
    # ==============================

    df["MarketAdjustedDemand"] = (
        df["Lag_1"] *
        (1 + df["IndustryGrowth"]/100) *
        (1 + df["SeasonIndex"]*0.05)
    )

    # ==============================
    # 7️⃣ SUPPLY SIMULATION
    # ==============================

    df["SupplyConstraint"] = np.random.uniform(0.85, 1.0, len(df))

    return df