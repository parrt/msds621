import numpy as np
from sklearn.datasets import load_boston, load_iris, load_wine, load_breast_cancer, fetch_california_housing
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from dtree import ClassifierTree621, RegressionTree621


"""
def test_toy():
    beds = [3, 2, 1, 1, 4, 4, 2, 1, 1, 4]
    baths = [1.5, 1., 1., 1., 1., 2., 1., 2., 1., 2.]
    price = [3000, 5465, 2850, 3275, 3350, 7995, 3600, 5645, 1725, 5800]
    X = np.stack((beds, baths), axis=1)
    y = np.array(price)
    run_regression_test(X, y, ntrials=10)
"""

def test_boston():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, ntrials=10, grace=0.11)

def test_boston_min_samples_leaf():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, ntrials=10, min_samples_leaf=3, grace=0.09, training_accuracy=.94)

def test_california_housing():
    X, y = fetch_california_housing(return_X_y=True)
    run_regression_test(X, y, ntrials=10, grace=0.19)

def test_iris():
    X, y = load_iris(return_X_y=True)
    run_classification_test(X, y, ntrials=10, grace=0.06)

def test_wine():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=10, grace=0.05)

def test_wine_min_samples_leaf():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=10, min_samples_leaf=3, grace=0.2)

def test_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    run_classification_test(X, y, ntrials=5, grace=0.05)


def run_regression_test(X, y, ntrials=5, min_samples_leaf=1, training_accuracy=0.90, grace=.08):
    idx = np.random.randint(0,len(X),size=min(len(X),500))
    X = X[idx]
    y = y[idx]
    scores = []
    train_scores = []
    sklearn_scores = []
    sklearn_train_scores = []
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.15)

        dt = RegressionTree621(min_samples_leaf=min_samples_leaf)
        dt.fit(X_train, y_train)
        score = dt.score(X_train, y_train)
        train_scores.append(score)
        score = dt.score(X_test, y_test)
        scores.append(score)

        sklearn_dt = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, max_features=1.0)
        sklearn_dt.fit(X_train, y_train)
        sklearn_score = sklearn_dt.score(X_train, y_train)
        sklearn_train_scores.append(sklearn_score)
        sklearn_score = sklearn_dt.score(X_test, y_test)
        sklearn_scores.append(sklearn_score)

    print()
    print(f"621     Train R^2 score mean {np.mean(train_scores):.2f}, stddev {np.std(train_scores):3f}")
    print(f"Sklearn Train R^2 score mean {np.mean(sklearn_train_scores):.2f}, stddev {np.std(sklearn_train_scores):3f}")
    print(f"621     Test  R^2 score mean {np.mean(scores):.2f}, stddev {np.std(scores):3f}")
    print(f"Sklearn Test  R^2 score mean {np.mean(sklearn_scores):.2f}, stddev {np.std(sklearn_scores):3f}")

    assert np.mean(train_scores)>=training_accuracy, \
           f"Training R^2: {np.mean(train_scores):.2f} must {training_accuracy:.2f}"
    assert np.mean(scores)+grace >= np.mean(sklearn_scores), \
           f"Testing R^2: {np.mean(scores):.2f} must be within {grace:.2f} of sklearn score: {np.mean(sklearn_scores):.2f}"


def run_classification_test(X, y, ntrials=5, min_samples_leaf=1, training_accuracy=0.95, grace=.05):
    X = X[:500]
    y = y[:500]
    scores = []
    train_scores = []
    sklearn_scores = []
    sklearn_train_scores = []
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.20)

        dt = ClassifierTree621(min_samples_leaf=min_samples_leaf)
        dt.fit(X_train, y_train)
        score = dt.score(X_train, y_train)
        train_scores.append(score)
        score = dt.score(X_test, y_test)
        scores.append(score)

        sklearn_dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_features=1.0)
        sklearn_dt.fit(X_train, y_train)
        sklearn_score = sklearn_dt.score(X_train, y_train)
        sklearn_train_scores.append(sklearn_score)
        sklearn_score = sklearn_dt.score(X_test, y_test)
        sklearn_scores.append(sklearn_score)

    print()
    print(f"621 accuracy score {np.mean(train_scores):.2f}, {np.mean(scores):.2f}")
    print(f"Sklearn accuracy score {np.mean(sklearn_train_scores):.2f}, {np.mean(sklearn_scores):.2f}")

    assert np.mean(train_scores)>=training_accuracy, \
           f"Training accuracy: {np.mean(train_scores):.2f} must {training_accuracy:.2f}"
    assert np.mean(scores)+grace >= np.mean(sklearn_scores), \
           f"Testing accuracy: {np.mean(scores):.2f} must be within {grace:.2f} of sklearn score: {np.mean(sklearn_scores):.2f}"
