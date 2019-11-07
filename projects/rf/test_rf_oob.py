import numpy as np
from sklearn.datasets import \
    load_boston, load_iris, load_diabetes, load_wine, \
    load_breast_cancer, fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

from rf_oob import RandomForestRegressor621, RandomForestClassifier621


def test_boston_oob():
    X, y = load_boston(return_X_y=True)
    run_regression_test(X, y, min_training_score = .86)

def test_diabetes_oob():
    X, y = load_diabetes(return_X_y=True)
    run_regression_test(X, y, min_training_score = .85)

def test_california_oob():
    X, y = fetch_california_housing(return_X_y=True)
    run_regression_test(X, y, min_training_score = .92)


def test_iris_oob():
    X, y = load_iris(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score = .95)

def test_wine_oob():
    X, y = load_wine(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score = .95)

def test_breast_cancer_oob():
    X, y = load_breast_cancer(return_X_y=True)
    run_classification_test(X, y, ntrials=5, min_training_score = .95)

def run_regression_test(X, y, ntrials=2, min_training_score = .90):
    X = X[:500]
    y = y[:500]
    scores = []
    train_scores = []
    oob_scores = []
    sklearn_scores = []
    sklearn_train_scores = []
    sklearn_oob_scores = []
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.20)

        rf = RandomForestRegressor621(n_trees=15, min_samples_leaf=3)
        rf.fit(X_train, y_train)
        score = rf.score(X_train, y_train)
        train_scores.append(score)
        score = rf.score(X_test, y_test)
        scores.append(score)
        print(f"RF R^2 OOB {rf.oob_score_}")
        oob_scores.append(rf.oob_score_)

        sklearn_rf = RandomForestRegressor(n_estimators=15, min_samples_leaf=3, max_features='sqrt', oob_score=True)
        sklearn_rf.fit(X_train, y_train)
        print(f"SKLEARN RF R^2 OOB {sklearn_rf.oob_score_}")
        sklearn_score = sklearn_rf.score(X_train, y_train)
        sklearn_train_scores.append(sklearn_score)
        sklearn_score = sklearn_rf.score(X_test, y_test)
        sklearn_scores.append(sklearn_score)
        sklearn_oob_scores.append(sklearn_rf.oob_score_)

    assert np.abs(np.mean(oob_scores)- np.mean(sklearn_oob_scores)) < 0.08, \
           f"OOB R^2: {np.mean(oob_scores):.2f} must be within 0.08 of sklearn score: {np.mean(sklearn_oob_scores):.2f}"

    assert np.mean(train_scores) >= min_training_score, \
           f"Training R^2: {np.mean(train_scores):.2f} must be >= {min_training_score}"
    assert np.mean(scores)+0.10 >= np.mean(sklearn_scores), \
           f"Testing R^2: {np.mean(scores):.2f} must be within 0.10 of sklearn score: {np.mean(sklearn_scores):.2f}"
    print(f"621 R^2 score {np.mean(train_scores):.2f}, {np.mean(scores):.2f}")
    print(f"Sklearn R^2 score {np.mean(sklearn_train_scores):.2f}, {np.mean(sklearn_scores):.2f}")

def run_classification_test(X, y, ntrials=1, min_training_score = .95):
    X = X[:500]
    y = y[:500]
    scores = []
    train_scores = []
    oob_scores = []
    sklearn_scores = []
    sklearn_train_scores = []
    sklearn_oob_scores = []
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.20)

        rf = RandomForestClassifier621(n_trees=20, min_samples_leaf=3)
        rf.fit(X_train, y_train)
        score = rf.score(X_train, y_train)
        train_scores.append(score)
        score = rf.score(X_test, y_test)
        scores.append(score)
        print(f"RF R^2 OOB {rf.oob_score_}")
        oob_scores.append(rf.oob_score_)

        sklearn_rf = RandomForestClassifier(n_estimators=20, min_samples_leaf=3, max_features='sqrt', oob_score=True)
        sklearn_rf.fit(X_train, y_train)
        print(f"SKLEARN RF R^2 OOB {sklearn_rf.oob_score_}")
        sklearn_score = sklearn_rf.score(X_train, y_train)
        sklearn_train_scores.append(sklearn_score)
        sklearn_score = sklearn_rf.score(X_test, y_test)
        sklearn_scores.append(sklearn_score)
        sklearn_oob_scores.append(sklearn_rf.oob_score_)

    assert np.abs(np.mean(scores)- np.mean(sklearn_scores)) < 0.05, \
           f"OOB R^2: {np.mean(oob_scores):.2f} must be within 0.05 of sklearn score: {np.mean(sklearn_oob_scores):.2f}"

    assert np.mean(train_scores) > min_training_score, \
           f"Training accuracy: {np.mean(train_scores):.2f} must {min_training_score:.2f}"
    assert np.mean(scores)+0.05 >= np.mean(sklearn_scores), \
           f"Testing accuracy: {np.mean(scores):.2f} must be within 0.05 of sklearn score: {np.mean(sklearn_scores):.2f}"
    print(f"621 accuracy score {np.mean(train_scores):.2f}, {np.mean(scores):.2f}")
    print(f"Sklearn accuracy score {np.mean(sklearn_train_scores):.2f}, {np.mean(sklearn_scores):.2f}")
