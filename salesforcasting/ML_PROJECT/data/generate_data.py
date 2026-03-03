import numpy as np
import pandas as pd

np.random.seed(42)

# ==============================
# CONFIGURATION
# ==============================

NUM_PRODUCTS = 120
NUM_QUARTERS = 12

product_ids = [f"P{i:03d}" for i in range(1, NUM_PRODUCTS + 1)]
quarters = list(range(1, NUM_QUARTERS + 1))

# ==============================
# 1️⃣ PRODUCT TABLE
# ==============================

products = pd.DataFrame({
    "ProductID": product_ids,
    "LaunchQuarter": np.random.randint(1, 6, NUM_PRODUCTS),
    "RAM": np.random.choice([4, 6, 8, 12], NUM_PRODUCTS),
    "Storage": np.random.choice([64, 128, 256], NUM_PRODUCTS),
    "ProcessorScore": np.random.randint(5, 10, NUM_PRODUCTS),
    "CameraMP": np.random.randint(12, 108, NUM_PRODUCTS),
    "Battery": np.random.randint(3000, 6000, NUM_PRODUCTS),
    "Is5G": np.random.choice([0, 1], NUM_PRODUCTS),
    "PremiumMaterial": np.random.choice([0, 1], NUM_PRODUCTS)
})

# ==============================
# 2️⃣ MARKET TABLE
# ==============================

market = pd.DataFrame({
    "Quarter": quarters,
    "GDPGrowth": np.random.uniform(2, 8, NUM_QUARTERS),
    "IndustryGrowth": np.random.uniform(5, 15, NUM_QUARTERS),
    "SeasonIndex": np.random.choice([0, 1], NUM_QUARTERS),
    "5GPenetration": np.linspace(10, 80, NUM_QUARTERS)
})

# ==============================
# 3️⃣ SALES + PRICING + SENTIMENT
# ==============================

records = []

for product in product_ids:
    for quarter in quarters:
        units_sold = np.random.randint(2000, 60000)

        launch_price = np.random.randint(15000, 80000)
        current_price = launch_price * np.random.uniform(0.7, 1)

        marketing_spend = np.random.randint(100000, 1000000)

        avg_rating = np.random.uniform(3, 5)
        review_volume = np.random.randint(100, 5000)
        sentiment_score = np.random.uniform(-1, 1)

        records.append([
            product,
            quarter,
            units_sold,
            launch_price,
            current_price,
            marketing_spend,
            avg_rating,
            review_volume,
            sentiment_score
        ])

sales_data = pd.DataFrame(records, columns=[
    "ProductID",
    "Quarter",
    "UnitsSold",
    "LaunchPrice",
    "CurrentPrice",
    "MarketingSpend",
    "AvgRating",
    "ReviewVolume",
    "SentimentScore"
])

# ==============================
# 4️⃣ JOIN ALL TABLES
# ==============================

full_data = (
    sales_data
    .merge(products, on="ProductID")
    .merge(market, on="Quarter")
)

# ==============================
# SAVE DATA
# ==============================

products.to_csv("products.csv", index=False)
market.to_csv("market.csv", index=False)
sales_data.to_csv("sales_data.csv", index=False)
full_data.to_csv("full_data.csv", index=False)

print("Enterprise dataset generated successfully.")