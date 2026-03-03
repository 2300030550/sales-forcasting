import pickle
import pandas as pd
import os
from portfolio_forecast import forecast_portfolio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def production_recommendation(quarter, safety_buffer=0.1):

    portfolio_df = forecast_portfolio(quarter)

    # ==================================
    # Safety Buffer (Enterprise logic)
    # ==================================

    portfolio_df["RecommendedProduction"] = (
        portfolio_df["ForecastedUnits"] *
        (1 + safety_buffer)
    )

    # ==================================
    # Production Adjustment Logic
    # ==================================

    portfolio_df["ProductionAction"] = "Maintain"

    portfolio_df.loc[
        portfolio_df["GrowthVsLastQuarter"] > 15,
        "ProductionAction"
    ] = "Increase Production"

    portfolio_df.loc[
        portfolio_df["GrowthVsLastQuarter"] < -10,
        "ProductionAction"
    ] = "Reduce Production"

    # ==================================
    # Capacity Stress Simulation
    # ==================================

    MAX_CAPACITY = 80000  # simulate per product constraint

    portfolio_df["CapacityAlert"] = portfolio_df[
        "RecommendedProduction"
    ] > MAX_CAPACITY

    return portfolio_df[
        [
            "ProductID",
            "ForecastedUnits",
            "RecommendedProduction",
            "ProductionAction",
            "CapacityAlert",
            "RiskCategory",
            "ForecastedRevenue",
            "ForecastedProfit"
        ]
    ]