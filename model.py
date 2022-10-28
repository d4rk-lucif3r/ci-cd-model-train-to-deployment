from luciferml.supervised.regression import Regression
import pandas as pd

def train_model():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/d4rk-lucif3r/LuciferML/master/examples/Salary_Data.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    regressor = Regression(
        predictor=["rfr"],
        cv_folds=10,
    )
    regressor.fit(X, y)
    return regressor.regressor, X, y

