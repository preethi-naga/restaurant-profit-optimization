from src.preprocess import load_and_clean, feature_engineering
from src.train import train_model

print("🔄 Starting project...")

# Step 1: Load data
df = load_and_clean("data/SkyCity Auckland Restaurants & Bars.csv")


print("✅ Data loaded")

# Step 2: Feature engineering
df = feature_engineering(df)
print("✅ Features created")

# Step 3: Train model
model = train_model(df)

print("✅ DONE: Model saved in models/model.pkl")