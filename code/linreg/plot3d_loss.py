import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # required even though not ref'd!
from matplotlib import rcParams
import matplotlib as mpl
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
    load_breast_cancer, load_diabetes, fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import glob
import os
from PIL import Image as PIL_Image


def np_normalize(x):
    return (x - np.mean(x)) / np.std(x)


def loss(B, X, y):
    "Line coefficients: B = [y-intercept, slope]"
    return np.mean(y - np.dot(X, np.array(B))) ** 2


def get_surface(X, y, loss, b0_range, b1_range):
    n = len(X)
    X = np.hstack([np.ones(shape=(n, 1)), X])  # add ones column
    L = np.zeros(shape=(len(b0_range),len(b1_range)))

    for x in range(len(b0_range)):
        for y in range(len(b1_range)):
            L[x][y] = loss([b0_range[x], b1_range[y]], X=X, y=y)
    return L


def plot3d(L, b0_range, b1_range, ax, elev=50, azim=145):
    rcParams["font.size"] = 10
    ax.view_init(elev, azim)

    b0_range_mesh, b1_range_mesh = np.meshgrid(b0_range, b1_range, indexing='xy')
    surface = ax.plot_surface(b0_range_mesh, b1_range_mesh, L.T, alpha=0.7, cmap='coolwarm')

    #    plt.title("""$loss(\\beta) = \sum_{i=1}^{N}(y^{{(i)}} - (\\beta_0 + \\beta_1 x^{{(i)}}))^2$""", fontsize=12)
    ax.set_xlabel('$\\beta_0$', fontsize=14)
    ax.set_ylabel('$\\beta_1$', fontsize=14)
    ax.zaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))


def data():
    n = 10
    df = pd.DataFrame()
    df['x'] = np.linspace(0, 10, num=n)
    df['y'] = df['x'] + 100 + np.random.normal(0, 1, size=n)

    X, y = df.drop('y', axis=1), df['y']
    return df, X, y


def cheese(norm = False):
    df = pd.read_csv('cheese_deaths.csv')
    X, y = df.cheese, df.deaths

    if norm:
        X = np_normalize(X)
    X = X.values
    X = X.reshape(-1, 1)
    y = y.values

    lm = LinearRegression()
    lm.fit(X, y)
    true_b0 = lm.intercept_
    true_b1 = lm.coef_[0]
    print(f"True b0 = {lm.intercept_:.2f}, b1 = {lm.coef_[0]:.1f}")

    if norm:
        b0_range = np.arange(0, 1000, 5)  # y intercept
        b1_range = np.arange(-450, 450, 5)  # slope
    else:
        b0_range = np.arange(-3030, -2900, 1)  # y intercept
        b1_range = np.arange(105, 120, .05)  # slope
    L = get_surface(X, y, loss, b0_range=b0_range, b1_range=b1_range)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    plot3d(L, b0_range=b0_range, b1_range=b1_range, ax=ax, elev=25, azim=0)
    ax.plot([true_b0], [true_b1], marker='x', markersize=10, color='black')
    plt.show()


def synthetic(norm = True):
    df, X, y = data()
    y.iloc[-1] *= 10  # add outlier

    # df = pd.read_csv('cheese_deaths.csv')
    # X, y = df.cheese, df.deaths

    if norm:
        X = np_normalize(X)
    X = X.values
    X = X.reshape(-1, 1)
    y = y.values

    lm = LinearRegression()
    lm.fit(X, y)
    true_b0 = lm.intercept_
    true_b1 = lm.coef_[0]
    print(f"OLS True b0 = {lm.intercept_:.2f}, b1 = {lm.coef_[0]:.1f}")

    lm = Ridge(alpha=10)
    lm.fit(X, y)
    ridge_true_b0 = lm.intercept_
    ridge_true_b1 = lm.coef_[0]
    print(f"Ridge True b0 = {lm.intercept_:.2f}, b1 = {lm.coef_[0]:.1f}")

    if norm:
        b0_range = np.arange(0, 300, 5)  # y intercept
        b1_range = np.arange(0, 160, 5)  # slope
    else:
        b0_range = np.arange(-50, 50, 1)  # y intercept
        b1_range = np.arange(0, 60, 1)  # slope
    L = get_surface(X, y, loss, b0_range=b0_range, b1_range=b1_range)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    plot3d(L, b0_range=b0_range, b1_range=b1_range, ax=ax, elev=25, azim=0)
    ax.plot([true_b0], [true_b1], marker='x', markersize=10, color='black')
    ax.plot([ridge_true_b0], [ridge_true_b1], marker='x', markersize=10, color='green')
    plt.show()


synthetic(norm=False)
#cheese()