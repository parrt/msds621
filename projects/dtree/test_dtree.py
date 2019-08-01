import numpy as np
from sklearn.datasets import \
    load_boston, load_iris, load_diabetes, load_wine, \
    load_breast_cancer, fetch_california_housing
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from dtree import ClassifierTree621, RegressionTree621


def test_boston():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, ntrials=10)

def test_california_housing():
    X, y = fetch_california_housing(return_X_y=True)
    run_regression_test(X, y, ntrials=10)


def test_iris():
    X, y = load_iris(return_X_y=True)
    run_classification_test(X, y, ntrials=10)

def test_wine():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=10)

def test_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    run_classification_test(X, y, ntrials=5)


def run_regression_test(X, y, ntrials=5):
    X = X[:500]
    y = y[:500]
    scores = []
    train_scores = []
    sklearn_scores = []
    sklearn_train_scores = []
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.15)

        dt = RegressionTree621()
        dt.fit(X_train, y_train)
        score = dt.score(X_train, y_train)
        train_scores.append(score)
        score = dt.score(X_test, y_test)
        scores.append(score)

        sklearn_dt = DecisionTreeRegressor(min_samples_leaf=1, max_features=1.0)
        sklearn_dt.fit(X_train, y_train)
        sklearn_score = sklearn_dt.score(X_train, y_train)
        sklearn_train_scores.append(sklearn_score)
        sklearn_score = sklearn_dt.score(X_test, y_test)
        sklearn_scores.append(sklearn_score)

    assert np.isclose(np.mean(train_scores), 1.0), \
           f"Training R^2: {np.mean(train_scores):.2f} must 1.0"
    assert np.mean(scores)+0.14 >= np.mean(sklearn_scores), \
           f"Testing R^2: {np.mean(scores):.2f} must be within 0.14 of sklearn score: {np.mean(sklearn_scores):.2f}"
    # print(f"621 R^2 score {np.mean(train_scores):.2f}, {np.mean(scores):.2f}")
    # print(f"Sklearn R^2 score {np.mean(sklearn_train_scores):.2f}, {np.mean(sklearn_scores):.2f}")

def run_classification_test(X, y, ntrials=5):
    X = X[:500]
    y = y[:500]
    scores = []
    train_scores = []
    sklearn_scores = []
    sklearn_train_scores = []
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.20)

        dt = ClassifierTree621()
        dt.fit(X_train, y_train)
        score = dt.score(X_train, y_train)
        train_scores.append(score)
        score = dt.score(X_test, y_test)
        scores.append(score)

        sklearn_dt = DecisionTreeClassifier(max_features=1.0)
        sklearn_dt.fit(X_train, y_train)
        sklearn_score = sklearn_dt.score(X_train, y_train)
        sklearn_train_scores.append(sklearn_score)
        sklearn_score = sklearn_dt.score(X_test, y_test)
        sklearn_scores.append(sklearn_score)

    assert np.isclose(np.mean(train_scores), 1.0), \
           f"Training accuracy: {np.mean(train_scores):.2f} must 1.0"
    assert np.mean(scores)+0.05 >= np.mean(sklearn_scores), \
           f"Testing accuracy: {np.mean(scores):.2f} must be within 0.05 of sklearn score: {np.mean(sklearn_scores):.2f}"
    # print(f"621 accuracy score {np.mean(train_scores):.2f}, {np.mean(scores):.2f}")
    # print(f"Sklearn accuracy score {np.mean(sklearn_train_scores):.2f}, {np.mean(sklearn_scores):.2f}")
