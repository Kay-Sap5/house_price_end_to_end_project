from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

models = {
    "LinearRegression":LinearRegression(),
    "SVR":SVR(),
    "RandomForest":RandomForestRegressor(),
    "DecisionTree":DecisionTreeRegressor(),
    "AdaBoostRegressor":AdaBoostRegressor(),
    "GradientBoost":GradientBoostingRegressor(),
    "Xgboost":XGBRegressor()
}

params = {

    "LinearRegression": {
        "fit_intercept": [True, False],
        "positive": [True, False]
    },

    "SVR": {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1],
        "gamma": ["scale", "auto"]
    },

    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },

    "DecisionTree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    },

    "AdaBoostRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0]
    },

    "GradientBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0]
    },

    "Xgboost": {
        "n_estimators": [100, 300],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
}
