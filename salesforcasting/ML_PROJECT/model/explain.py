import shap
import pickle
import pandas as pd
import os

# ==============================
# LOAD MODEL + FEATURES
# ==============================

model = pickle.load(open("model_p50.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# Load some sample data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "full_data.csv")

df = pd.read_csv(data_path)

from utils.feature_pipeline import build_features
df = build_features(df)

X = df[feature_columns]

# ==============================
# SHAP EXPLAINER
# ==============================

explainer = shap.Explainer(model)
shap_values = explainer(X)

# ==============================
# GLOBAL FEATURE IMPORTANCE
# ==============================

shap.summary_plot(shap_values, X)