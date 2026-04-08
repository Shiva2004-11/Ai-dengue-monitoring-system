import pandas as pd

def load_data():

    features = pd.read_csv("dataset/dengue_features_train.csv")
    labels = pd.read_csv("dataset/dengue_labels_train.csv")

    df = pd.merge(features, labels, on=["city","year","weekofyear"])

    df = df.ffill()

    return df


def prepare_features(df):

    selected_features = [
        "reanalysis_air_temp_k",
        "reanalysis_specific_humidity_g_per_kg",
        "precipitation_amt_mm",
        "station_avg_temp_c"
    ]

    X = df[selected_features]
    y = df["total_cases"]

    return X, y