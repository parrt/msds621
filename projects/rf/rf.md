#  Random Forests

## Goal

The goal of this project is to leverage the decision tree implementation from the previous project to make a random forest implementation. The goal is to build the simplest possible functional random forest without concern for efficiency but with accuracy comparable to sklearn. You will create objects `RandomForestRegressor621` and `RandomForestClassifier621` as drop in replacements for sklearn's implementations. My implementation is about 100 lines of code, but minor changes are also required to the decision tree implementation in `dtree.py`.

Your implementation must include code to support out-of-bag (OOB) validation error estimation. It's a bit tricky to get right, so the OOB unit tests are worth less in my evaluation. You can still get 92% total, even if you don't implement OOB error estimation.

You will work in git repo `rf`-*userid* and create `rf.py` in the root directory of the repo. Also copy your `dtree.py` from the previous project into the root directory, as you will need to modify it and use it for this project.

## Description

Classification and regression trees do an excellent job of fitting a model to the training data. Unfortunately, decision trees are a little too good and they overfit like mad, meaning that they do not generalize well to previously-unseen test data. To increase generality, random forests use a collection of decision trees that have been weakened to make them more independent. We trade a bit of accuracy for dramatically improved generality.

A random forest does not feed all data to every decision tree in its collection. Each tree is trained on a bootstrapped version of the original training set. Further, RFs must sometimes forget some of the available features during training. In our case, decision node splitting will be limited to considering a random selection of features of size `max_features`, a hyper parameter not used in our decision trees. Naturally, both bootstrapping and setting a maximum features per split introduce noise into the predictions of the individual decision trees. But, averaging results of these tree estimators squeezes the noise back down. We get the best of both worlds!

### Bootstrapping

The goal of bootstrapping for random forests is to train a number of decision trees that are as independent and identically distributed as possible by using different but similar training sets.  Each tree trains on a slightly different subset of the training data. Bootstrapping, in theory, asks the underlying distribution that generated the data to generate another independent sample. In practice, bootstrapping gets about 2/3 of the X rows, leaving 1/3 "out of bag" (OOB). See [sklearn's resample function](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html) for a handy way to get a list of indexes to help create a bootstrap sample training set. For example, if I have a numpy array with a list of indexes in `idx` from `X`, then `X[idx]`  is a list of rows from 2D matrix `X`.

The algorithm for fitting a random forest is then:

<img src="images/fit.png" width="50%">


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
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        ...
```

A `LeafNode` obviously just returns itself (`self`) rather than the prediction.

The second change is in the training mechanism. The  decision tree for looks like:

<img src="images/dtreefit.png" width="50%">

For fitting conventional decision trees, `bestsplit()` exhaustively scans all available features and the feature values looking for the optimal variable/split combination. To reduce overfitting, each split should pick from a random subset of the features; the subset size is the hyper perimeter `max_features`.  Function `np.random.choice()` is useful here to get a list of feature indexes and then `X[:, i]` gives us the ith column.  In my solution,  the change is to the outermost loop in `find_best_split()`:

<img src="images/bestsplit.png" width="60%">

### RF Prediction

Once we've trained a forest of decision trees, we can make predictions for one or more feature vectors using `predict()`. For regression, the prediction for the forest is the weighted average of the predictions from the individual decision trees. If `X_test` passed to `predict()` is a two-dimensional matrix of *n* rows, then *n* predictions should be returned in an array from `predict()`. To make a prediction for a single feature vector, call `leaf()` on each tree to get the leaf node that contains the prediction information for the feature vector.  Each leaf has `n`, the number of observations in that leaf that can serve as our weight. The leaf also has a `prediction` that is the predicted y value for regression or class for classification. (We'll deal with classification separately.) First, compute the total weight of all `leaf.n` and then compute the sum of `leaf.prediction * leaf.n`. The prediction is then the weighted sum divided by the total weight.

For classification, it's a little more complicated because we need a majority vote across all trees.  As with regression, go through all of the trees, and get the leaves associated with the prediction of a single feature vector.  Create a numpy array, say, `class_counts` that is big enough so that the maximum integer representing a class is a valid index in the array. Then, increment `class_counts[y]` for each `y` in each leaf associated with `x` test vector.  Here are the algorithms:

<img src="images/predict-regr.png" width="60%">

<img src="images/predict-class.png" width="40%">

###  Regressor and classifier class definitions

To mimic sklearn machine learning models, we need to create some class definitions. You are free to implement the regression and classifier tree objects as you like, but you must satisfy the appropriate interface so that the unit tests will run.  Here is my setup:

<img src="images/hierarchy.png" width="60%">

The `RandomForest621` class has my generic `fit()` method that is inherited by subclasses `RandomForest Regressor621` and `RandomForestClassifier621`.  Field `n_estimators` is the number of trees in the forest and I compute/store the number of unique `y` values in `nunique` because, for classification, we need to know how many classes there are.

Method `compute_oob_score()` is just a helper method that I used to encapsulate that functionality, but you can do whatever you want. `RandomForest621.fit()` calls  `self.compute_oob_score()` and that calls the implementation either in regressor or classifier, depending on which object I created.

You can use the following class definitions as templates:

```
class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        ...
        if self.oob_score:
            self.oob_score_ = ... compute OOB score ...
```

```
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = ...

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        ...
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        ...
```

```
class RandomForestClassifier621:
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.trees = ...

    def predict(self, X_test) -> np.ndarray:
        ...
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        ...    
```

## Getting started

Download the [test scripts](https://github.com/parrt/msds621/tree/master/projects/rf) and create blank script file `rf.py` with perhaps `import numpy as np` as starter code.  Copy your `dtree.py` script from the previous project into this new `rf-`*userid* project directory as well.  Make sure that you are creating files in the root directory of the repository. Add the files to your repository, commit, and push back to github.

In this way, you have started on the project without actually having to do any work. Getting over inertia is an important step in any project.

## Out-of-bag (OOB) error

The R^2 and accuracy scores for OOB observations is an accurate estimate of the validation error, all without having to manually hold out a validation or test set. This is a major advantage of random forests.

A bootstrapped sample is roughly 2/3 of the training records for any given tree, which leaves 1/3 of the samples (OOB) as test set. After training each decision tree, keep track of the OOB records in the tree.  For example, I do `t.oob_idxs = ...` inside my `fit()` method (for each tree `t`).  After training all trees in `fit()`, loop through the trees again and compute the OOB score, if hyperparameter `self.oob_score` is true. Save the score in `self.oob_score_` for either the RF regressor or classifier object, which is consistent with the sklearn implementation. See the class lecture slides for more details, but here are the algorithms again:

<img src="images/oob-score-regr.png" width="60%">

<img src="images/oob-score-class.png" width="60%">

### Speed issues

Make sure to avoid loops and use vector operations. For example, don't loop through OOB sample indexes, use the list of indexes as an index into NumPy array:

```
a = np.arange(10,20)
idx = [1,3,5]
print(a[idx]) # gives [11, 13, 15]
```

To find all array values greator than 0, use:

```
nonzero_idx = np.where(a > 0)
```


## Deliverables

In your github repo `rf`-*userid*, you must provide the following files at the root of the repository directory:

* `dtree.py` This is the code from your previous project  but with the updates specified above to randomly select from a subset of the features during each split.
* `rf.py` This is file containing your `RandomForestRegressor621` and `RandomForestClassifier621` implementations, and any other functions or classes you need.

I will copy in a clean version of the test script before grading your projects.

## Evaluation

Because these tests take so long and they are completely independent, we can test a number of them in parallel to speed things up. 

```
$ pip install pytest-xdist
```

For me, it's 3x faster when I use `-n 8` option (on my 4-core fast intel i7 CPU): `pytest -v -n 8 test_rf.py`.  For clarity, though, here is the output I get from PyCharm (using just one processor, not 8):

```
============================= test session starts ==============================
platform darwin -- Python 3.7.1, pytest-5.3.0, py-1.7.0, pluggy-0.13.0 -- /Users/parrt/anaconda3/bin/python
cachedir: .pytest_cache
rootdir: /Users/parrt/courses/msds621-private/projects/rf
plugins: xdist-1.30.0, doctestplus-0.2.0, arraydiff-0.3, remotedata-0.3.1, openfiles-0.3.1, forked-1.1.3
collecting ... collected 26 items

test_rf.py::test_boston PASSED                                           [  3%]
boston: 621     Train R^2 score mean 0.92, stddev 0.008252
boston: Sklearn Train R^2 score mean 0.94, stddev 0.002283
boston: 621     Test  R^2 score mean 0.83, stddev 0.039512
boston: Sklearn Test  R^2 score mean 0.87, stddev 0.016250

test_rf.py::test_boston_oob PASSED                                       [  7%]
boston_oob: 621 OOB score 0.79 vs sklearn OOB 0.81
boston_oob: 621     Train R^2 score mean 0.90, stddev 0.017967
boston_oob: Sklearn Train R^2 score mean 0.92, stddev 0.000181
boston_oob: 621     Test  R^2 score mean 0.85, stddev 0.029832
boston_oob: Sklearn Test  R^2 score mean 0.84, stddev 0.050604

test_rf.py::test_boston_min_samples_leaf PASSED                          [ 11%]
boston_min_samples_leaf: 621     Train R^2 score mean 0.89, stddev 0.024142
boston_min_samples_leaf: Sklearn Train R^2 score mean 0.88, stddev 0.007884
boston_min_samples_leaf: 621     Test  R^2 score mean 0.84, stddev 0.050672
boston_min_samples_leaf: Sklearn Test  R^2 score mean 0.82, stddev 0.051951

test_rf.py::test_boston_all_features PASSED                              [ 15%]
boston_all_features: 621     Train R^2 score mean 0.94, stddev 0.003891
boston_all_features: Sklearn Train R^2 score mean 0.94, stddev 0.001431
boston_all_features: 621     Test  R^2 score mean 0.83, stddev 0.048552
boston_all_features: Sklearn Test  R^2 score mean 0.84, stddev 0.069147

test_rf.py::test_boston_most_features PASSED                             [ 19%]
boston_most_features: 621     Train R^2 score mean 0.94, stddev 0.002705
boston_most_features: Sklearn Train R^2 score mean 0.95, stddev 0.002711
boston_most_features: 621     Test  R^2 score mean 0.84, stddev 0.045269
boston_most_features: Sklearn Test  R^2 score mean 0.85, stddev 0.036696

test_rf.py::test_boston_min_samples_leaf_oob PASSED                      [ 23%]
boston_min_samples_leaf_oob: 621 OOB score 0.80 vs sklearn OOB 0.80
boston_min_samples_leaf_oob: 621     Train R^2 score mean 0.89, stddev 0.011317
boston_min_samples_leaf_oob: Sklearn Train R^2 score mean 0.90, stddev 0.004134
boston_min_samples_leaf_oob: 621     Test  R^2 score mean 0.78, stddev 0.060680
boston_min_samples_leaf_oob: Sklearn Test  R^2 score mean 0.77, stddev 0.050456

test_rf.py::test_diabetes PASSED                                         [ 26%]
diabetes: 621     Train R^2 score mean 0.76, stddev 0.003718
diabetes: Sklearn Train R^2 score mean 0.79, stddev 0.008023
diabetes: 621     Test  R^2 score mean 0.47, stddev 0.035870
diabetes: Sklearn Test  R^2 score mean 0.50, stddev 0.052362

test_rf.py::test_diabetes_ntrees PASSED                                  [ 30%]
diabetes_ntrees: 621     Train R^2 score mean 0.77, stddev 0.000719
diabetes_ntrees: Sklearn Train R^2 score mean 0.80, stddev 0.003353
diabetes_ntrees: 621     Test  R^2 score mean 0.44, stddev 0.054706
diabetes_ntrees: Sklearn Test  R^2 score mean 0.48, stddev 0.057942

test_rf.py::test_diabetes_all_features PASSED                            [ 34%]
diabetes_all_features: 621     Train R^2 score mean 0.80, stddev 0.008528
diabetes_all_features: Sklearn Train R^2 score mean 0.81, stddev 0.010541
diabetes_all_features: 621     Test  R^2 score mean 0.57, stddev 0.024089
diabetes_all_features: Sklearn Test  R^2 score mean 0.55, stddev 0.050219

test_rf.py::test_diabetes_most_features PASSED                           [ 38%]
diabetes_most_features: 621     Train R^2 score mean 0.80, stddev 0.003375
diabetes_most_features: Sklearn Train R^2 score mean 0.81, stddev 0.003624
diabetes_most_features: 621     Test  R^2 score mean 0.42, stddev 0.016757
diabetes_most_features: Sklearn Test  R^2 score mean 0.47, stddev 0.009754

test_rf.py::test_diabetes_oob PASSED                                     [ 42%]
diabetes_oob: 621 OOB score 0.37 vs sklearn OOB 0.36
diabetes_oob: 621     Train R^2 score mean 0.77, stddev 0.012186
diabetes_oob: Sklearn Train R^2 score mean 0.78, stddev 0.002872
diabetes_oob: 621     Test  R^2 score mean 0.44, stddev 0.046698
diabetes_oob: Sklearn Test  R^2 score mean 0.48, stddev 0.010731

test_rf.py::test_california_housing PASSED                               [ 46%]
california_housing: 621     Train R^2 score mean 0.89, stddev 0.006762
california_housing: Sklearn Train R^2 score mean 0.88, stddev 0.007321
california_housing: 621     Test  R^2 score mean 0.81, stddev 0.015160
california_housing: Sklearn Test  R^2 score mean 0.80, stddev 0.006308

test_rf.py::test_california_housing_oob PASSED                           [ 50%]
california_housing_oob: 621 OOB score 0.74 vs sklearn OOB 0.75
california_housing_oob: 621     Train R^2 score mean 0.89, stddev 0.009270
california_housing_oob: Sklearn Train R^2 score mean 0.90, stddev 0.000961
california_housing_oob: 621     Test  R^2 score mean 0.81, stddev 0.038924
california_housing_oob: Sklearn Test  R^2 score mean 0.79, stddev 0.041484

test_rf.py::test_iris_ntrees PASSED                                      [ 53%]
iris_ntrees: 621 accuracy score 0.97, 0.95
iris_ntrees: Sklearn accuracy score 0.98, 0.95

test_rf.py::test_iris PASSED                                             [ 57%]
iris: 621 accuracy score 0.96, 0.92
iris: Sklearn accuracy score 0.98, 0.93

test_rf.py::test_iris_all_features PASSED                                [ 61%]
iris_all_features: 621 accuracy score 0.97, 0.93
iris_all_features: Sklearn accuracy score 0.98, 0.93

test_rf.py::test_iris_most_features PASSED                               [ 65%]
iris_most_features: 621 accuracy score 0.97, 0.95
iris_most_features: Sklearn accuracy score 0.97, 0.95

test_rf.py::test_iris_oob PASSED                                         [ 69%]
iris_oob: 621 OOB score 0.94 vs sklearn OOB 0.94
iris_oob: 621 accuracy score 0.97, 0.91
iris_oob: Sklearn accuracy score 0.98, 0.93

test_rf.py::test_wine PASSED                                             [ 73%]
wine: 621 accuracy score 0.99, 0.94
wine: Sklearn accuracy score 1.00, 0.97

test_rf.py::test_wine_all_features PASSED                                [ 76%]
wine_all_features: 621 accuracy score 1.00, 0.97
wine_all_features: Sklearn accuracy score 0.99, 0.98

test_rf.py::test_wine_most_features PASSED                               [ 80%]
wine_most_features: 621 accuracy score 0.99, 0.96
wine_most_features: Sklearn accuracy score 1.00, 0.97

test_rf.py::test_wine_oob PASSED                                         [ 84%]
wine_oob: 621 OOB score 0.93 vs sklearn OOB 0.94
wine_oob: 621 accuracy score 0.99, 0.98
wine_oob: Sklearn accuracy score 0.99, 0.98

test_rf.py::test_wine_min_samples_leaf PASSED                            [ 88%]
wine_min_samples_leaf: 621 accuracy score 1.00, 0.98
wine_min_samples_leaf: Sklearn accuracy score 0.99, 0.97

test_rf.py::test_wine_min_samples_leaf_oob PASSED                        [ 92%]
wine_min_samples_leaf_oob: 621 OOB score 0.94 vs sklearn OOB 0.93
wine_min_samples_leaf_oob: 621 accuracy score 0.99, 0.97
wine_min_samples_leaf_oob: Sklearn accuracy score 0.99, 0.97

test_rf.py::test_breast_cancer PASSED                                    [ 96%]
breast_cancer: 621 accuracy score 0.97, 0.95
breast_cancer: Sklearn accuracy score 0.99, 0.95

test_rf.py::test_breast_cancer_oob PASSED                                [100%]
breast_cancer_oob: 621 OOB score 0.94 vs sklearn OOB 0.94
breast_cancer_oob: 621 accuracy score 0.97, 0.94
breast_cancer_oob: Sklearn accuracy score 0.99, 0.94
=============================== warnings summary ===============================
...
================= 26 passed, 3 warnings in 181.61s (0:03:01) ==================
```

PyCharm knows how to do this as well, if you look at the configurations and add `-n 6` or `-n 8` as an additional argument to run six unit tests at once::

<img src="images/pycharm-pytest.png" width="50%">

There are 8 OOB tests and each failed test costs you 1%, for total of 92% maximum if you don't implement this functionality.

The other unit tests check basic regression classification but also try out combinations of `max_features`, `min_samples_leaf`, `n_estimators`.   For the non-OOB tests, each failed test cost you 5%.

I also have created a hidden test on a different data set and failing it costs 14% of your grade.

*My test passes in roughly 1 minute for test_rf, running in parallel with -n 8, and you will lose 10% if all tests takes longer than about 2 minutes total.*

### Automatic testing using github actions

As with the previous projects, I have provided a [Github actions](https://docs.github.com/en/free-pro-team@latest/actions) file for you to get automatic testing. All you have to do is put the [test.yml](https://github.com/parrt/msds621/blob/master/projects/rf/test.yml) file I have prepared for you into repo subdirectory `.github/workflows`, commit, and push back to github. Then go to the Actions tab of your repository.

Naturally it will only work if you have your software written and added to the repository. Once you have something basic working, this functionality is very nice because it automatically shows you how your software is going to run on a different computer (a linux computer). This will catch the usual errors where you have hardcoded something from your machine into the software. It also gets you in the habit of committing software to the repository as you develop it, rather than using the repository as a homework submission device.

If you want to get fancy, you can use the following "badge" code in your repo README.md file:

```
Test rf

![rf](https://github.com/parrt/parrt-rf/workflows/Test%20MSDS621%20rf/badge.svg)
```