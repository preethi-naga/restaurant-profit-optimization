import pandas as pd

def predict_profit(model, input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]