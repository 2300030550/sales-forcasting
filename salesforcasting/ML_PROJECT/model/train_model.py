import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.feature_pipeline import build_features


# ==============================
# 1️⃣ LOAD DATA
# ==============================

df = pd.read_csv("../data/full_data.csv")
# Apply feature engineering
df = build_features(df)

# ==============================
# 2️⃣ DEFINE TARGET & FEATURES
# ==============================

target = "UnitsSold"

feature_columns = [
    col for col in df.columns
    if col not in ["UnitsSold", "ProductID"]
]

X = df[feature_columns]
y = df[target]

# ==============================
# 3️⃣ TIME-AWARE SPLIT
# ==============================

df = df.sort_values(["ProductID", "Quarter"])

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ==============================
# 4️⃣ BASE MODEL (P50 Forecast)
# ==============================

model_p50 = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_p50.fit(X_train, y_train)

# ==============================
# 5️⃣ EVALUATION
# ==============================

y_pred = model_p50.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# ==============================
# 6️⃣ QUANTILE MODELS (UNCERTAINTY)
# ==============================

# ==============================
# 6️⃣ TRUE QUANTILE MODELS
# ==============================

model_p10 = XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.1,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

model_p90 = XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.9,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

model_p10.fit(X_train, y_train)
model_p90.fit(X_train, y_train)

# ==============================
# 7️⃣ SAVE ARTIFACTS
# ==============================

pickle.dump(model_p50, open("model_p50.pkl", "wb"))
pickle.dump(model_p10, open("model_p10.pkl", "wb"))
pickle.dump(model_p90, open("model_p90.pkl", "wb"))
pickle.dump(feature_columns, open("feature_columns.pkl", "wb"))

print("Enterprise models saved successfully.")