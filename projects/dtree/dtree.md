#  Classifier and regressor decision trees

## Goal

Your goal is to implement decision trees for classification and regression. You will make two similar implementations, first as a small set of functions and then as a objects similar to sklearn's `DecisionTreeClassifier` and `DecisionTreeRegressor`.  This project can be challenging because you must recursively construct trees, but my solution is only 100 lines of Python with comments.

You will work in git repo `dtree`-*userid*.

## Description

We will learn how to build a decision trees as part of our lectures ...

Script `test_dtree_funcs.py` tests your implementation.

After successfully building the functions that construct trees and make predictions, the next phase is to wrap or pull apart these functions and wrap them into class definitions: `RegressionTree621` and `ClassifierTree621` to mimic sklearn's `DecisionTreeClassifier` and `DecisionTreeRegressor` objects. Script `test_dtree.py` tests your implementation.

```
class RegressionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini
    def fit(self, X, y):
        ...
    def predict(self, X_test):
        ...        
    def score(self, X_test, y_test):
        ...
```

```
class ClassifierTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini
    def fit(self, X, y):
        ...
    def predict(self, X_test):
        ...        
    def score(self, X_test, y_test):
        ...
```

If you have a strong programming background, you are welcome to use class inheritance to inherit and/or override methods. In that case you might use something like:

```
class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini
    ...

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)
    ...

class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)
    ...
```

## Getting started

## Deliverables

```
class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        ...
```

```
class LeafNode:
    def __init__(self, y, prediction):
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        ...
```

```
def predict(root, X_test):
    return np.array([root.predict(X_test[ri,:]) for ri in range(len(X_test))])
```

```
def regressor_fit(X, y, min_samples_leaf=1):
    return fit(X, y, isclassifier=False, min_samples_leaf=min_samples_leaf, loss=np.std)
```

```
def classifier_fit(X, y, min_samples_leaf=1):
    return fit(X, y, isclassifier=True, min_samples_leaf=min_samples_leaf, loss=gini)
```

```
def fit(X, y, isclassifier, min_samples_leaf=1, loss=None):
    """
    Recursively create and return a decision tree fit to (X,y) for
    either a classifier or regressor. Leaf nodes for classifiers predict
    the most common class (the mode) and regressors predict the average y
    for samples in that leaf.

    The loss function is either np.std (if isclassifier) or gini.
    """
```

```
def gini(x):
    "Return the gini impurity score for values in x"
```

## Evaluation

```
$ python -m pytest -v test_dtree_funcs.py 
============================================== test session starts ===============================================
platform darwin -- Python 3.7.1, pytest-4.0.2, py-1.7.0, pluggy-0.8.0 -- /Users/parrt/anaconda3/bin/python
cachedir: .pytest_cache
rootdir: ...
plugins: remotedata-0.3.1, openfiles-0.3.1, doctestplus-0.2.0, arraydiff-0.3
collected 5 items                                                                                                

test_dtree_funcs.py::test_boston PASSED                                                                    [ 20%]
test_dtree_funcs.py::test_california_housing PASSED                                                        [ 40%]
test_dtree_funcs.py::test_iris PASSED                                                                      [ 60%]
test_dtree_funcs.py::test_wine PASSED                                                                      [ 80%]
test_dtree_funcs.py::test_breast_cancer PASSED                                                             [100%]

=========================================== 5 passed in 26.71 seconds ============================================
```