import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)

    # Rename for consistency
    df.rename(columns={'DeliveryCostPerOrder': 'DeliveryCostOrder'}, inplace=True)

    # Clean
    df.ffill(inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def feature_engineering(df):

    # Total Net Profit
    df['TotalNetProfit'] = (
        df['InStoreNetProfit'] +
        df['UberEatsNetProfit'] +
        df['DoorDashNetProfit'] +
        df['SelfDeliveryNetProfit']
    )

    # Profit per order
    df['ProfitPerOrder'] = df['TotalNetProfit'] / df['MonthlyOrders']

    # Interaction features
    df['UE_Impact'] = df['CommissionRate'] * df['UE_share']
    df['SD_CostImpact'] = df['DeliveryCostOrder'] * df['SD_share']

    return df