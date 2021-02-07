import numpy as np
import pandas as pd
from scipy.special import lmbda

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from linreg import *


def addnoise(X):
    df = pd.DataFrame(data=X)
    for i in range(10):
        df[f'ID{i}'] = range(1, len(X) + 1)
    return df.values

def MAE(a,b): return np.mean(np.abs(a.flatten()-b.flatten()))

def synthetic_data():
    n = 1000
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
    X = pd.get_dummies(X)
    y = df_ames['SalePrice']
    y = y.values.reshape(-1, 1)
    return X, y

def check(X, y, mae, model, skmodel, r2_diff):
    normalize(X)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.10, shuffle=True)#, random_state=999)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print("r^2", r2)

    estimated_B = model.B.reshape(-1)

    skmodel.fit(X_train, y_train.reshape(-1,1))
    y_pred = skmodel.predict(X_test)
    skr2 = r2_score(y_test.reshape(-1,1), y_pred)
    print("sklearn r^2", skr2)
    if skmodel.coef_.ndim==2:
        true_B = np.concatenate([skmodel.intercept_, skmodel.coef_[0]])
    else:
        true_B = np.concatenate([skmodel.intercept_, skmodel.coef_])
    # print(estimated_B, 'VS\n', true_B)
    # print(np.std(estimated_B), np.std(true_B), MAE(estimated_B, true_B))

    # sklearn comes back with insanely big coeff sometimes for test_boston_noise
    # after index 13 so snips those off
    estimated_B = estimated_B[0:13]
    true_B = true_B[0:13]

    # COMPARE COEFF
    r = pd.DataFrame()
    r['estimated'] = estimated_B
    r['true'] = true_B
    print(r)

    print(f'MAE of coefficients difference {MAE(estimated_B, true_B)}')

    assert np.abs(r2-skr2) < r2_diff, f"R^2 {r2} and sklearn R^2 {skr2} differ by {np.abs(r2-skr2)}"
    assert MAE(estimated_B, true_B) < mae, f'MAE of coefficients difference {MAE(estimated_B, true_B)} >= {mae}'


def test_synthetic():
    X, y = synthetic_data()

    check(X, y, .0005,
          LinearRegression621(eta=1, max_iter=60_000),
          LinearRegression(),
          r2_diff=0.002)


def test_ridge_synthetic():
    X, y = synthetic_data()

    check(X, y, .12,
          RidgeRegression621(max_iter=100_000, eta=5, lmbda=80),
          Ridge(alpha=80, solver='lsqr'),
          r2_diff=0.6)


def test_boston():
    boston = load_boston()
    X = boston.data
    y = boston.target
    y = y.reshape(-1, 1)

    check(X, y, .003,
          LinearRegression621(max_iter=30_000, eta=5),
          LinearRegression(),
          r2_diff=0.0001)

def test_boston_noise():
    boston = load_boston()
    X = boston.data
    y = boston.target
    y = y.reshape(-1, 1)

    X = addnoise(X)

    check(X, y, .28,
          LinearRegression621(max_iter=15_000, eta=5),
          LinearRegression(),
          r2_diff=0.3)

def test_ridge_boston():
    boston = load_boston()
    X = boston.data
    y = boston.target
    y = y.reshape(-1, 1)

    check(X, y, 1.1,
          RidgeRegression621(max_iter=30_000, eta=.1, lmbda=70),
          Ridge(alpha=70, solver='lsqr'),
          r2_diff=0.2)

def test_ridge_boston_noise():
    boston = load_boston()
    X = boston.data
    y = boston.target
    y = y.reshape(-1, 1)

    X = addnoise(X)

    check(X, y, .65,
          RidgeRegression621(max_iter=30_000, eta=5, lmbda=80),
          Ridge(alpha=80, solver='lsqr'),
          r2_diff=0.2)
