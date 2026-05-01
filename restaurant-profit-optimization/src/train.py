import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

def train_model(df):

    features = [
        'InStoreShare',
        'UE_share',
        'DD_share',
        'SD_share',
        'CommissionRate',
        'DeliveryCostOrder',
        'DeliveryRadiusKM',
        'GrowthFactor'
    ]

    X = df[features]
    y = df['TotalNetProfit']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained successfully!")

    return model
