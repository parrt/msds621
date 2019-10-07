import numpy as np
import pandas as pd
from scipy.special import lmbda

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
                             load_breast_cancer, load_diabetes, fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, r2_score

import inspect

from linreg import *


def addnoise(X):
    df = pd.DataFrame(data=X)
    for i in range(10):
        df[f'ID{i}'] = range(1, len(X) + 1)
    return df.values

def MAE(a,b): return np.mean(np.abs(a.flatten()-b.flatten()))

def synthetic_data():
    n = 10
    df = pd.DataFrame()
    df['x'] = np.linspace(0, 10, num=n)
    df['y'] = df['x'] + np.random.normal(0, 1, size=n)
    X = df['x'].values
    y = df['y'].values
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    y[-1] = 100  # create an outlier
    return X, y

def load_ames():
    df_ames = pd.read_csv('https://raw.githubusercontent.com/parrt/msds621/master/data/ames.csv')  # 1460 records
    df_ames = df_ames.sample(n=300, replace=False)
    print(len(df_ames), "rows")
    cols_with_missing = df_ames.columns[df_ames.isnull().any()]
    cols = set(df_ames.columns) - set(cols_with_missing)
    X = df_ames[cols]
    X = X.drop('SalePrice', axis=1)
    normalize(X)
    X = pd.get_dummies(X)
    y = df_ames['SalePrice']
    y = y.values.reshape(-1, 1)
    return X, y

def check(X, y, mae, model, skmodel, donormalize=True):
    if donormalize:
        normalize(X)

    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    # print("r^2", r2)

    estimated_B = model.B.reshape(-1)

    skmodel.fit(X, y)
    y_pred = skmodel.predict(X)
    skr2 = r2_score(y, y_pred)
    # print("sklearn r^2", skr2)
    if skmodel.coef_.ndim==2:
        true_B = np.concatenate([skmodel.intercept_, skmodel.coef_[0]])
    else:
        true_B = np.concatenate([skmodel.intercept_, skmodel.coef_])
    # print(estimated_B, 'VS\n', true_B)
    # print(np.std(estimated_B), np.std(true_B), MAE(estimated_B, true_B))

    # COMPARE COEFF
    # r = pd.DataFrame()
    # r['estimated'] = estimated_B
    # r['true'] = true_B
    # print(r)

    # print(f'MAE of coefficients difference >= {MAE(estimated_B, true_B)}')

    assert np.abs(r2-skr2) < .07, f"R^2 {r2} and sklearn R^2 {skr2} differ by {np.abs(r2-skr2)}"
    assert MAE(estimated_B, true_B) < mae, f'MAE of coefficients difference >= {mae}'


def test_synthetic():
    X, y = synthetic_data()

    check(X, y, .0005,
          LinearRegression621(eta=1, max_iter=60_000),
          LinearRegression())


def test_ridge_synthetic():
    X, y = synthetic_data()

    check(X, y, .00005,
          RidgeRegression621(max_iter=30_000, eta=1, lmbda=4),
          Ridge(alpha=40, solver='lsqr'))


def test_boston():
    boston = load_boston()
    X = boston.data
    y = boston.target
    y = y.reshape(-1, 1)

    check(X, y, .003,
          LinearRegression621(max_iter=15_000, eta=.8),
          LinearRegression())

def test_boston_noise():
    boston = load_boston()
    X = boston.data
    y = boston.target
    y = y.reshape(-1, 1)

    X = addnoise(X)

    check(X, y, .28,
          LinearRegression621(max_iter=15_000, eta=1),
          LinearRegression())

def test_ridge_boston():
    boston = load_boston()
    X = boston.data
    y = boston.target
    y = y.reshape(-1, 1)

    check(X, y, .06,
          RidgeRegression621(max_iter=30_000, eta=1, lmbda=.1),
          Ridge(alpha=40, solver='lsqr'))

def test_ridge_boston_noise():
    boston = load_boston()
    X = boston.data
    y = boston.target
    y = y.reshape(-1, 1)

    X = addnoise(X)

    check(X, y, .05,
          RidgeRegression621(max_iter=30_000, eta=1, lmbda=.1),
          Ridge(alpha=40, solver='lsqr'))
