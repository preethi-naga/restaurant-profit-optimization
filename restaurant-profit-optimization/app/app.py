import streamlit as st
import pickle
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # ✅ Fix for deployment
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 🎨 UI STYLING
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f5f7fa, #c3cfe2);
}
h1 { text-align: center; color: #2c3e50; }
h2, h3 { color: #34495e; }
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🔷 LOAD MODEL
# =========================
model_path = os.path.join(os.path.dirname(__file__), "../models/model.pkl")
model = pickle.load(open(model_path, "rb"))

# ✅ Feature order fix
model_features = model.feature_names_in_

# =========================
# 🔷 SAFE PREDICT FUNCTION
# =========================
def predict(df):
    try:
        df = df.reindex(columns=model_features, fill_value=0)
        return model.predict(df)[0]
    except:
        return 0

# =========================
# 🔷 PAGE CONFIG
# =========================
st.set_page_config(page_title="Restaurant Profit Optimizer", layout="centered")

st.title("🍽️ Restaurant Profit Optimization Dashboard")
st.caption("👉 Adjust sliders → Click Predict → Analyze results")

# =========================
# 🔷 INPUTS
# =========================
st.subheader("📊 Channel Mix")

ue = st.slider("Uber Eats Share", 0.0, 1.0, 0.3)
dd = st.slider("DoorDash Share", 0.0, 1.0, 0.2)
sd = st.slider("Self Delivery Share", 0.0, 1.0, 0.2)

total = ue + dd + sd

if total > 1:
    st.error("⚠️ Total share cannot exceed 1")
    st.stop()

instore = 1 - total
st.write(f"In-store Share: {instore:.2f}")

st.subheader("💰 Cost Parameters")

commission = st.slider("Commission Rate", 0.0, 0.5, 0.25)
delivery_cost = st.slider("Delivery Cost", 0.5, 6.0, 2.0)
radius = st.slider("Delivery Radius", 3, 20, 10)
growth = st.slider("Growth Factor", 0.9, 1.1, 1.0)

# =========================
# 🔷 INPUT DATA
# =========================
input_data = {
    'InStoreShare': instore,
    'UE_share': ue,
    'DD_share': dd,
    'SD_share': sd,
    'CommissionRate': commission,
    'DeliveryCostOrder': delivery_cost,
    'DeliveryRadiusKM': radius,
    'GrowthFactor': growth
}

input_df = pd.DataFrame([input_data])

# =========================
# 🔷 PREDICTION
# =========================
if st.button("💰 Predict Profit"):

    profit = predict(input_df)
    st.success(f"Predicted Profit: ₹ {profit:.2f}")

    # Range
    st.info(f"Range: ₹ {profit*0.9:.2f} to ₹ {profit*1.1:.2f}")

    # =========================
    # KPIs
    # =========================
    st.subheader("📊 KPIs")

    sensitivity = commission * 100
    efficiency = profit / (total + 0.01)
    breakeven = max(0.05, 0.35 - sd * 0.1)

    baseline = pd.DataFrame([{
        'InStoreShare': 0.5,
        'UE_share': 0.2,
        'DD_share': 0.2,
        'SD_share': 0.1,
        'CommissionRate': 0.25,
        'DeliveryCostOrder': 2,
        'DeliveryRadiusKM': 10,
        'GrowthFactor': 1.0
    }])

    baseline_profit = predict(baseline)
    uplift = ((profit - baseline_profit) / (abs(baseline_profit)+1))*100

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    c1.metric("Profit", f"{profit:.2f}")
    c2.metric("Sensitivity", f"{sensitivity:.2f}")
    c3.metric("Efficiency", f"{efficiency:.2f}")
    c4.metric("Break-even", f"{breakeven:.2f}")

    st.metric("Uplift %", f"{uplift:.2f}%")

    # =========================
    # GRAPH 1
    # =========================
    st.subheader("📈 Profit vs Commission")

    commissions = np.linspace(0.05, 0.5, 10)
    profits = []

    for c in commissions:
        temp = input_data.copy()
        temp['CommissionRate'] = c
        profits.append(predict(pd.DataFrame([temp])))

    plt.figure()
    plt.plot(commissions, profits)
    plt.axhline(y=0)
    plt.grid()
    st.pyplot(plt)

    # =========================
    # GRAPH 2
    # =========================
    st.subheader("📉 Profit vs Delivery Cost")

    costs = np.linspace(0.5, 6, 10)
    cost_profits = []

    for dc in costs:
        temp = input_data.copy()
        temp['DeliveryCostOrder'] = dc
        cost_profits.append(predict(pd.DataFrame([temp])))

    plt.figure()
    plt.plot(costs, cost_profits)
    plt.axhline(y=0)
    plt.grid()
    st.pyplot(plt)

    # =========================
    # OPTIMIZATION
    # =========================
    st.subheader("🎯 Recommendation")

    if commission > 0.3:
        st.warning("Reduce commission")
    elif sd < 0.2:
        st.info("Increase self-delivery")
    elif instore < 0.3:
        st.warning("Improve in-store sales")
    else:
        st.success("Strategy is good")

    # =========================
    # BEST STRATEGY
    # =========================
    st.subheader("🔍 Best Strategy")

    best_profit = -1e9
    best_config = None

    for ue_i in np.linspace(0.1, 0.5, 5):
        for sd_i in np.linspace(0.1, 0.4, 4):

            dd_i = 0.2
            instore_i = 1 - (ue_i + sd_i + dd_i)

            if instore_i < 0:
                continue

            temp = {
                'InStoreShare': instore_i,
                'UE_share': ue_i,
                'DD_share': dd_i,
                'SD_share': sd_i,
                'CommissionRate': commission,
                'DeliveryCostOrder': delivery_cost,
                'DeliveryRadiusKM': radius,
                'GrowthFactor': growth
            }

            pred = predict(pd.DataFrame([temp]))

            if pred > best_profit:
                best_profit = pred
                best_config = temp

    st.success(f"Best Profit: {best_profit:.2f}")
    st.write(best_config)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("✔ Predictive Modeling & Profit Optimization")