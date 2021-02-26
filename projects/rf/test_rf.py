import numpy as np
from sklearn.datasets import \
    load_boston, load_iris, load_diabetes, load_wine, \
    load_breast_cancer, fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import inspect

from rf import RandomForestRegressor621, RandomForestClassifier621


def test_boston():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, min_training_score = .86)

def test_boston_oob():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, min_training_score = .86, oob=True)

def test_boston_min_samples_leaf():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, ntrials=5, min_samples_leaf=5, grace=0.08)

def test_boston_all_features():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, ntrials=3, max_features=1.0, grace=0.08)

def test_boston_most_features():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, ntrials=4, max_features=2/3, grace=0.08)

def test_boston_min_samples_leaf_oob():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, ntrials=5, min_samples_leaf=5, grace=0.08, oob=True)

def test_diabetes():
    X, y = load_diabetes(return_X_y=True)
    run_regression_test(X, y, min_training_score = .72, grace=0.11)

def test_diabetes_ntrees():
    X, y = load_diabetes(return_X_y=True)
    run_regression_test(X, y, min_training_score = .72, grace=0.11, n_estimators=25)

def test_diabetes_all_features():
    X, y = load_diabetes(return_X_y=True)
    run_regression_test(X, y, min_training_score = .72, grace=0.11, max_features=1.0)

def test_diabetes_most_features():
    X, y = load_diabetes(return_X_y=True)
    run_regression_test(X, y, min_training_score = .72, grace=0.11, max_features=2/3)

def test_diabetes_oob():
    X, y = load_diabetes(return_X_y=True)
    run_regression_test(X, y, min_training_score = .72, grace=0.11, oob=True)

def test_california_housing():
    X, y = fetch_california_housing(return_X_y=True)
    run_regression_test(X, y, min_training_score = .79, grace=0.15)

def test_california_housing_oob():
    X, y = fetch_california_housing(return_X_y=True)
    run_regression_test(X, y, min_training_score = .79, grace=0.15, oob=True)


def test_iris_ntrees():
    X, y = load_iris(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.93, n_estimators=25)

def test_iris():
    X, y = load_iris(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.93)

def test_iris_all_features():
    X, y = load_iris(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.93, max_features=1.0)

def test_iris_most_features():
    X, y = load_iris(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.93, max_features=2/3)

def test_iris_oob():
    X, y = load_iris(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.93, oob=True)

def test_wine():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.98)

def test_wine_all_features():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.98, max_features=1.0)

def test_wine_most_features():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.98, max_features=2/3)

def test_wine_oob():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.98, oob=True)

def test_wine_min_samples_leaf():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=10, min_training_score=0.98, min_samples_leaf=5, grace=0.2)

def test_wine_min_samples_leaf_oob():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=10, min_training_score=0.98, min_samples_leaf=5, grace=0.2, oob=True)

def test_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.96)

def test_breast_cancer_oob():
    X, y = load_breast_cancer(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score=0.96, oob=True)


def run_regression_test(X, y, ntrials=2, min_training_score = .85, min_samples_leaf=3, max_features=0.3, grace=.08, oob=False, n_estimators=18):
    stack = inspect.stack()
    caller_name = stack[1].function[len('test_'):]
    X = X[:500]
    y = y[:500]

    test_scores = []
    train_scores = []
    oob_scores = []

    sklearn_scores = []
    sklearn_train_scores = []
    sklearn_oob_scores = []

    for i in range(ntrials):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.20)

        rf = RandomForestRegressor621(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob)
        rf.fit(X_train, y_train)
        train_scores.append(rf.score(X_train, y_train))
        test_scores.append(rf.score(X_test, y_test))
        oob_scores.append(rf.oob_score_)

        sklearn_rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob)
        sklearn_rf.fit(X_train, y_train)
        sklearn_train_scores.append(sklearn_rf.score(X_train, y_train))
        sklearn_scores.append(sklearn_rf.score(X_test, y_test))
        if oob:
            sklearn_oob_scores.append(sklearn_rf.oob_score_)
        else:
            sklearn_oob_scores.append(0.0)

    print()
    if oob:
        print(f"{caller_name}: 621 OOB score {np.mean(oob_scores):.2f} vs sklearn OOB {np.mean(sklearn_oob_scores):.2f}")
    print(f"{caller_name}: 621     Train R^2 score mean {np.mean(train_scores):.2f}, stddev {np.std(train_scores):3f}")
    print(f"{caller_name}: Sklearn Train R^2 score mean {np.mean(sklearn_train_scores):.2f}, stddev {np.std(sklearn_train_scores):3f}")
    print(f"{caller_name}: 621     Test  R^2 score mean {np.mean(test_scores):.2f}, stddev {np.std(test_scores):3f}")
    print(f"{caller_name}: Sklearn Test  R^2 score mean {np.mean(sklearn_scores):.2f}, stddev {np.std(sklearn_scores):3f}")

    assert np.mean(train_scores) >= min_training_score, \
           f"Training R^2: {np.mean(train_scores):.2f} must be >= {min_training_score}"
    assert np.mean(test_scores)+grace >= np.mean(sklearn_scores), \
           f"Testing R^2: {np.mean(test_scores):.2f} must be within {grace:.2f} of sklearn score: {np.mean(sklearn_scores):.2f}"
    if oob:
        assert np.abs(np.mean(oob_scores) - np.mean(sklearn_oob_scores)) < grace, \
            f"OOB R^2: {np.mean(oob_scores):.2f} must be within {grace:2f} of sklearn score: {np.mean(sklearn_oob_scores):.2f}"


def run_classification_test(X, y, ntrials=1, min_samples_leaf=3, max_features=0.3, min_training_score=1.0, grace=.07, oob=False, n_estimators=15):
    stack = inspect.stack()
    caller_name = stack[1].function[len('test_'):]
    X = X[:500]
    y = y[:500]

    test_scores = []
    train_scores = []
    oob_scores = []

    sklearn_test_scores = []
    sklearn_train_scores = []
    sklearn_oob_scores = []

    for i in range(ntrials):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.20)

        rf = RandomForestClassifier621(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob)
        rf.fit(X_train, y_train)
        train_scores.append(rf.score(X_train, y_train))
        test_scores.append(rf.score(X_test, y_test))
        oob_scores.append(rf.oob_score_)

        sklearn_rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob)
        sklearn_rf.fit(X_train, y_train)
        sklearn_train_scores.append(sklearn_rf.score(X_train, y_train))
        sklearn_test_scores.append(sklearn_rf.score(X_test, y_test))
        if oob:
            sklearn_oob_scores.append(sklearn_rf.oob_score_)
        else:
            sklearn_oob_scores.append(0.0)

    if oob:
        assert np.abs(np.mean(oob_scores) - np.mean(sklearn_oob_scores)) < grace, \
               f"OOB accuracy: {np.mean(oob_scores):.2f} must be within {grace:.2f} of sklearn score: {np.mean(sklearn_oob_scores):.2f}"
    assert np.mean(train_scores) >= min_training_score, \
           f"Training accuracy: {np.mean(train_scores):.2f} must {min_training_score:.2f}"
    assert np.mean(test_scores)+grace >= np.mean(sklearn_test_scores), \
           f"Testing accuracy: {np.mean(test_scores):.2f} must be within {grace:.2f} of sklearn score: {np.mean(sklearn_test_scores):.2f}"

    print()
    if oob:
        print(f"{caller_name}: 621 OOB score {np.mean(oob_scores):.2f} vs sklearn OOB {np.mean(sklearn_oob_scores):.2f}")
    print(f"{caller_name}: 621 accuracy score {np.mean(train_scores):.2f}, {np.mean(test_scores):.2f}")
    print(f"{caller_name}: Sklearn accuracy score {np.mean(sklearn_train_scores):.2f}, {np.mean(sklearn_test_scores):.2f}")
