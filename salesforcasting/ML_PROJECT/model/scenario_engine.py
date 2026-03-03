import pickle
import pandas as pd
import os
from utils.feature_pipeline import build_features

# Load model + features
model = pickle.load(open("model_p50.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "full_data.csv")

df = pd.read_csv(data_path)
df = build_features(df)

def simulate_scenario(product_id, quarter,
                      price_change_pct=0,
                      marketing_change_pct=0):

    # Select product-quarter row
    row = df[
        (df["ProductID"] == product_id) &
        (df["Quarter"] == quarter)
    ].copy()

    if row.empty:
        return "Product/Quarter not found"

    baseline_features = row[feature_columns]
    baseline_prediction = model.predict(baseline_features)[0]

    # Apply scenario changes
    row["CurrentPrice"] *= (1 + price_change_pct/100)
    row["MarketingSpend"] *= (1 + marketing_change_pct/100)

    row = build_features(row)
    scenario_features = row[feature_columns]
    scenario_prediction = model.predict(scenario_features)[0]

    impact = scenario_prediction - baseline_prediction

    return {
        "baseline_sales": int(baseline_prediction),
        "scenario_sales": int(scenario_prediction),
        "change_in_units": int(impact)
    }