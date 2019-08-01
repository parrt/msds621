#  Random Forests

## Goal

The goal of this project is to leverage the decision tree implementation from the previous project to make a random forest implementation. The goal is to build the simplest possible functional random forest without concern for efficiency or getting the best accuracy. You will create objects `RandomForestRegressor621` and `RandomForestClassifier621` as drop in replacements for sklearn's implementations. My implementation is about 60 lines of code but changes are also required to the decision tree implementation in `dtree.py`.

As a bonus for strong programmers, I provide a description of how to implement out-of-bag test error estimates (and a test script).

You will work in git repo `rf`-*userid*.

## Description

Classification and regression trees do an excellent job of fitting a model to the training data. Unfortunately, it is a little too good and they overfit like mad, meaning that they do not generalize well to previously-unseen test data. To increase generality, random forests use a collection of decision trees that have been weakened. We trade a bit of bias for dramatically improved generality.

A random forest does not feed all data to every decision tree in its collection. Further, it sometimes forgets about available features during training. In our case, decision node splitting will be limited to the considering a random selection of features of size "square root of the number of total features." Naturally, both of these introduce bias into the individual decision trees, but combining results of these trees brings the bias back down. We get the best of both worlds.

### Bootstrapping

The goal of bootstrapping for random forests is to train a number of uncorrelated decision trees on similar but different training data sets.  Each tree trains on a slightly different subset of the training data. Bootstrapping, in principle, asks the underlying distribution that generated the data to generate another independent sample. In practice, bootstrapping gets about 2/3 of the X rows, leaving 1/3 "out of bag" (OOB). See [sklearn's resample function](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html) for a handy way to get a list of indexes to help create a bootstrap sample training set. For example, if I have a numpy array with a list of indexes in `idx` from `X`, then `X[idx]`  is a list of rows from 2D matrix `X`.

The algorithm for fitting a random forest is then:

```
fit(X, y):
	for each tree t:
		X_, y_ = bootstrapped sample from X, y
		t.fit(X_, y_)
```

### Changes to decision tree training

There are two important changes we need to make to the decision tree mechanism so that it is suitable for random forests.

First, we need to update the interface for our decision nodes so that decision trees know how to return the leaf of the tree that should make the prediction, rather than the prediction itself as `predict()` does.   To produce a weighted average, we need to know not just the prediction, but also the number of samples within that leaf. (The `predict()` function only returns the prediction.)

```
class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild
    def predict(self, x_test):
        ...
    def leaf(self, x_test):
        "Like predict(), but returns the leaf node not the prediction"
        ...
```

You might find the Python conditional `isinstance(child, LeafNode)` of use. And answers whether or not the `child` object is a `LeafNode` as opposed to a decision node.

The second change is in the training mechanism. Conventional decision trees exhaustively scan all available features and the feature values looking for the optimal variable/split combination. To reduce overfitting, each split should pick from a random subset of the features; the subset size is the square root of the number of features.  Function `np.random.choice()` is useful here to get a list of feature indexes and then `X[:, i]` gives us the ith column.  In my solution,  the change is to the outermost loop in `find_best_split()`.

### RF Prediction

Once we've trained a forest of decision trees, we can make predictions for one or more feature vectors using `predict()`. For regression, the prediction for the forest is the weighted average of the predictions from the individual decision trees. If `X_test` passed to `predict()` is a two-dimensional matrix of *n* rows, then *n* predictions should be returned in an array from `predict()`. To make a prediction for a single feature vector, call `leaf()` on each tree to get the leaf node that contains the prediction information for the feature vector.  Each leaf has `n`, the number of observations in that leaf that can serve as our weight. The leaf also has a `prediction` that is the predicted y value for regression or class for classification. (We'll deal with classification separately.) First, compute the total weight of all `leaf.n` and then compute the sum of `leaf.prediction * leaf.n`. The prediction is then the weighted sum divided by the total weight.

For classification, it's a little more complicated Because we need a majority vote across all trees.  As with regression, go through all of the trees, and get the leaves associated with the prediction of a single feature vector.  Create a numpy array, say, `class_counts` that is big enough so that the maximum integer representing a class is a valid index in the array. Then, `class_counts[c]` gives the count associated with class `c`.  Add `leaf.n` to the `class_counts[leaf.prediction]` count. The class with the largest count should be the prediction; `np.argmax` is useful here.

### Object definitions

To mimic sklearn machine learning models, we need to create some class definitions. You can use the following as templates:

```
class RandomForestRegressor621:
    def __init__(self, n_trees=10, min_samples_leaf=3):
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.trees = ...

    def fit(self, X, y) -> None:
        ...

    def predict(self, X_test) -> np.ndarray:
        ...
        
    def leaf(self, x_test):
        "Like predict(), but returns the leaf node not the prediction"
        return self.root.leaf(x_test)

    def score(self, X_test, y_test) -> float:
        ...
```

and

```
class RandomForestClassifier621:
    def __init__(self, n_trees=10, min_samples_leaf=3):
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.trees = ...

    def fit(self, X, y) -> None:
        ...

    def predict(self, X_test) -> np.ndarray:
        ...
        
    def leaf(self, x_test):
        "Like predict(), but returns the leaf node not the prediction"
        return self.root.leaf(x_test)

    def score(self, X_test, y_test) -> float:
        ...    
```

## Getting started

Download the [test scripts](https://github.com/parrt/msds621/tree/master/projects/rf) and create blank script file `rf.py` with perhaps `import numpy as np` as starter code.  Copy your `dtree.py` script from the previous project into your project directory as well.  Make sure that you are creating files in the root directory of the repository. Add the files to your repository, commit, and push back to github.

In this way, you have started on the project without actually having to do any work. Getting over inertia is an important step in any project.

## Optional extension for out of bag error

If this project was too easy for you, and you would like more of a challenge, try to implement the OOB functionality. The R^2 and accuracy scores for OOB observations is and accurate estimate of the test error, all without requiring manual creation of a validation or test set. This is a major advantage of random forests.

A bootstrapped sample is roughly 2/3 of the training records for any given tree, which leaves 1/3 of the samples (OOB) as test set. After training each decision tree, make predictions for the OOB samples (those records not use to train that specific tree).  You will get a prediction for each OOB sample and you must add that to an overall prediction for that record. You must also track how many predictions were added so that you can compute an average later. Set field `self.oob_score_` to be consistent with the sklearn implementation.

For classification, you must track class counts for each OOB record with something like:

```
oob_sample_predictions = np.zeros(shape=(len(X),self.nunique))
```

Then, as with RF prediction, bump the count of the class predicted for each OOB record. Not every record from X will appear in some OOB sample because we're using a small number of trees like 15 or 20. Ignoring the zero counts, the OOB prediction should be the category with the largest count. That gives you the prediction for all X that appeared in some OOB sample. Test the accuracy of that with the associated y records. That is not the best OOB accuracy measure we can do, but it's close enough for our purposes.

I have provided a version of the tests for you so you can check your work:

```
$ python -m pytest -v test_rf_oob.py 
=============================== test session starts ================================
platform darwin -- Python 3.7.1, pytest-4.0.2, py-1.7.0, pluggy-0.8.0 -- ...
cachedir: .pytest_cache
rootdir: ...
plugins: remotedata-0.3.1, openfiles-0.3.1, doctestplus-0.2.0, arraydiff-0.3
collected 6 items                                                                  

test_rf_oob.py::test_boston_oob PASSED                                       [ 16%]
test_rf_oob.py::test_diabetes_oob PASSED                                     [ 33%]
test_rf_oob.py::test_california_oob PASSED                                   [ 50%]
test_rf_oob.py::test_iris_oob PASSED                                         [ 66%]
test_rf_oob.py::test_wine_oob PASSED                                         [ 83%]
test_rf_oob.py::test_breast_cancer_oob PASSED                                [100%]

================================= warnings summary =================================
...

-- Docs: https://docs.pytest.org/en/latest/warnings.html
====================== 6 passed, 3 warnings in 29.94 seconds =======================
```

## Deliverables

In your github repo `rf`-*userid*, you must provide the following files at the root of the repository directory:

* `dtree.py` This is the code from your previous project  but with the updates specified above to randomly select from a subset of the features during each split.
* `rf.py` This is file containing your `RandomForestRegressor621` and `RandomForestClassifier621` implementations.

I will copy in a clean version of the test script before grading your projects.

## Evaluation

We will run test script `test_rf.py` to evaluate your projects. With luck, getting a single test to pass for regression means that all regressors will pass. Getting a single classifier test to pass should mean you pass all of those. Regardless, you will receive 16.6% for each of 6 tests passed.

```
$ python -m pytest -v test_rf.py
=============================== test session starts ================================
platform darwin -- Python 3.7.1, pytest-4.0.2, py-1.7.0, pluggy-0.8.0 -- ...
cachedir: .pytest_cache
rootdir: ...
plugins: remotedata-0.3.1, openfiles-0.3.1, doctestplus-0.2.0, arraydiff-0.3
collected 6 items                                                                  

test_rf.py::test_boston PASSED                                               [ 16%]
test_rf.py::test_diabetes PASSED                                             [ 33%]
test_rf.py::test_california_housing PASSED                                   [ 50%]
test_rf.py::test_iris PASSED                                                 [ 66%]
test_rf.py::test_wine PASSED                                                 [ 83%]
test_rf.py::test_breast_cancer PASSED                                        [100%]

============================ 6 passed in 30.26 seconds =============================
```

*My test passes in roughly 40 seconds and you will lose 5% if your test takes longer than 80 seconds.*
