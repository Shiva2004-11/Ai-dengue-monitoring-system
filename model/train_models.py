import os
import sys
import pickle

# ensure project root is on sys.path so local packages like `utils` can be imported
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from utils.preprocessing import load_data, prepare_features

df = load_data()

X, y = prepare_features(df)

rf = RandomForestRegressor()
lr = LinearRegression()
xgb = XGBRegressor()

rf.fit(X,y)
lr.fit(X,y)
xgb.fit(X,y)

models = {
    "Random Forest": rf,
    "Linear Regression": lr,
    "XGBoost": xgb
}

pickle.dump(models, open("model/models.pkl","wb"))

print("Models trained successfully")