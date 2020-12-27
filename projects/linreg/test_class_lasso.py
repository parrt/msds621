# NOT REQUIRED but used to test optional LassoLogistic621

import numpy as np
import pandas as pd
from scipy.special import lmbda

np.random.seed(999) # Force same random sequence for each test

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from linreg import *

def MAE(a,b): return np.mean(np.abs(a-b))


def wine_data():
    wine = load_wine()
    df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df_wine['y'] = wine.target
    df_wine = df_wine[df_wine['y'] < 2]  # get two-class dataset
    X = df_wine.drop('y', axis=1).values
    y = df_wine['y'].values
    y = y.reshape(-1, 1)
    return X, y

def iris_data():
    iris = load_iris()

    df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df_iris['y'] = iris.target
    df_iris = df_iris[df_iris['y'] < 2]
    X = df_iris.drop('y', axis=1).values
    y = df_iris['y'].values
    y = y.reshape(-1, 1)
    return X, y

def check(X, y, mae, model, skmodel, accuracy=1.0):
    normalize(X)
    model.fit(X, y)
    y_pred = model.predict(X)
    correct = np.sum(y.flatten() == y_pred.flatten())
    n = len(X)
    # print(f"Got {correct} / {n} correct = {(correct / n) * 100:.2f}%")

    estimated_B = model.B.reshape(-1)
    # print(estimated_B)

    skmodel.fit(X, y.reshape(-1))
    if skmodel.coef_.ndim==2:
        true_B = np.concatenate([skmodel.intercept_, skmodel.coef_[0]])
    else:
        true_B = np.concatenate([skmodel.intercept_, skmodel.coef_])
    print("MAE of B", MAE(estimated_B, true_B))

    y_proba_estimated = model.predict_proba(X)
    y_proba_true = skmodel.predict_proba(X)
    print(f"Log loss {log_loss(y, y_proba_estimated)} vs sklearn {log_loss(y, y_proba_true)}")
    assert np.abs(log_loss(y, y_proba_estimated) - log_loss(y, y_proba_true)) < 0.001

    r = pd.DataFrame()
    r['estimated'] = estimated_B
    r['true'] = true_B
    print(r)

    assert (correct/n) >= accuracy
    assert MAE(estimated_B, true_B) < mae


def test_lasso_synthetic():
    # From https://beckernick.github.io/logistic-regression-from-scratch/
    n = 5000
    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], n)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], n)

    X = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((np.zeros(n), np.ones(n)))
    y = y.reshape(-1,1)

    check(X, y, .8,
          LassoLogistic621(max_iter=15_000, eta=4),
          LogisticRegression(penalty='l1', max_iter=15_000, solver='saga'),
          accuracy=.98)

def test_lasso_wine():
    X, y = wine_data()

    check(X, y, .07,
          LassoLogistic621(max_iter=15_000, lmbda=1, eta=6),
          LogisticRegression(penalty='l1', max_iter=5_000, solver='saga'))

def test_lasso_iris():
    X, y = iris_data()

    check(X, y, .03,
          LassoLogistic621(max_iter=20_000, lmbda=1, eta=5),
          LogisticRegression(penalty='l1', max_iter=5_000, solver='saga'))
