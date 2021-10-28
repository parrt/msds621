from sklearn.linear_model import LogisticRegression
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
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, shuffle=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    correct = np.sum(y_test.flatten() == y_pred.flatten())
    n = len(X_test)
    print(f"Got {correct} / {n} correct = {(correct / n) * 100:.2f}%")

    estimated_B = model.B.reshape(-1)
    # print(estimated_B)

    skmodel.fit(X_train, y_train.reshape(-1))

    if skmodel.coef_.ndim==2:
        true_B = np.concatenate([skmodel.intercept_, skmodel.coef_[0]])
    else:
        true_B = np.concatenate([skmodel.intercept_, skmodel.coef_])
    print("MAE of B", MAE(estimated_B, true_B))

    y_proba_estimated = model.predict_proba(X_test)
    y_proba_true = skmodel.predict_proba(X_test)
    print(f"Log loss {log_loss(y_test, y_proba_estimated)} vs sklearn {log_loss(y_test, y_proba_true)}")
    if log_loss(y_test, y_proba_estimated) > log_loss(y_test, y_proba_true):
        # Sometimes log loss is pretty far off despite accuracy being ok
        # depending on validation set; these are really small data sets
        assert np.abs(log_loss(y_test, y_proba_estimated) - log_loss(y_test, y_proba_true)) < 0.365

    r = pd.DataFrame()
    r['estimated'] = estimated_B
    r['true'] = true_B
    print(r)

    assert (correct/n) >= accuracy
    assert MAE(estimated_B, true_B) < mae


def test_synthetic():
    # From https://beckernick.github.io/logistic-regression-from-scratch/
    n = 5000
    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], n)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], n)

    X = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((np.zeros(n), np.ones(n)))
    y = y.reshape(-1,1)

    # X_ = X.copy()
    # normalize(X_)
    # logit = sm.Logit(y, add1col(X_))
    # res = logit.fit()
    # print(res.summary())
    # # print(logit.fit().params)

    check(X, y, .025,
          LogisticRegression621(max_iter=10_000, eta=10),
          LogisticRegression(penalty='none', solver='lbfgs'),
          accuracy=.98)

def test_wine():
    X, y = wine_data()

    check(X, y, 2.25,
          LogisticRegression621(max_iter=30_000, eta=1),
          LogisticRegression(penalty='none', solver='lbfgs'),
          accuracy=0.88)

def test_iris():
    X, y = iris_data()

    check(X, y, 1.8,
          LogisticRegression621(max_iter=80_000, eta=1),
          LogisticRegression(penalty='none', solver='lbfgs'),
          accuracy=0.99)
