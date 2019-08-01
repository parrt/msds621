#  Classifier and regressor decision trees

## Goal

Your goal is to implement decision trees for classification and regression. You will make two similar implementations, first as a small set of functions and then as a objects similar to sklearn's `DecisionTreeClassifier` and `DecisionTreeRegressor`.  This project can be challenging because you must recursively construct trees, but my solution is only 100 lines of Python with comments.

You will work in git repo `dtree`-*userid*.

## Description

We will learn how to build a decision trees as part of our lectures but here's a brief synopsis. Decision trees partition feature space into hyper volumes with similar features, subject to the goal of reducing variance in the target y variable. For example, a regression of y against a single feature begins by finding the split point in x that gets two regions with the lowest variance within those regions. This is done by exhaustively testing locations in x space, computing the variance of y for observations to the left and the variance of observations on the right of the split point. The location of the best average of these two variances is the split point. Then, the algorithm recursively splits the two new regions. Here's what it looks like after 1, 2, and 3 splits for regression:

<img src="images/cars-1.svg" width="30%"> <img src="images/cars-2.svg" width="30%"> <img src="images/cars-3.svg" width="30%">

The same process works for classification. The only difference is that, instead of measuring variance, we measure and try to reduce the uncertainty/purity ([gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)) of the y values to the left and right of the split.

<img src="images/iris-1.svg" width="30%"> <img src="images/iris-2.svg" width="30%"> <img src="images/iris-3.svg" width="30%">

If you prefer, you can also look at the partitions with stacked bar charts:
 
 <img src="images/iris-1-bar.svg" width="40%">

When moving beyond a single feature, during training, the decision tree must choose not only the split value but also the feature to test. Again, it exhaustively tries all combinations of features and feature values, looking for the lowest variance (regression) or highest purity (classification) to the left and right of the candidate split value. For example, here is what feature space looks like for 2 features from the mtcars data set:

<img src="https://user-images.githubusercontent.com/178777/49104999-4edb0d80-f234-11e8-9010-73b7c0ba5fb9.png" width="50%">

And, from the IRIS data set, here is what partitioning looks like for two of those features for classification purposes.

<img src="images/iris-2D-1.svg" width="40%"> <img src="images/iris-2D-2.svg" width="40%">

For 1 feature, we get ranges as hyper volumes. For 2 features, we get rectangular regions, and for 3 features we get rectangular volumes, etc... After that it's impossible to visualize but the same process holds. Partition the feature space into hyper volumes while trying to reduce variance or impurity in y after each split.

If you record the sequence of splits, you get a binary tree. For example, here is the classification tree for IRIS

<img src="https://github.com/parrt/dtreeviz/raw/master/testing/samples/iris-TD-2.svg?sanitize=true" width="40%">

### Functions and objects to build

First, define two classes that will represent the objects in your decision trees. You can build them anywhere you want, but here's the outline of how I built mine:

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

Please make sure, however, that your tree nodes respond to function `t.predict(x)` for some tree node `t` and feature vector `x`.

The primary interface to your code from the testing script is the `fit()` function:

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

For example, the test script has the following function for classification:

```
def classifier_fit(X, y, min_samples_leaf=1):
    return fit(X, y, isclassifier=True, min_samples_leaf=min_samples_leaf, loss=gini)
```

Note that it is passing in the `gini()` function (see below).

Next, you must define a function that takes a decision tree root and one or more feature vectors (in a 2D matrix) and returns one or more predictions: 

```
def predict(root, X_test):
    ...
```

The predictions are either numeric values for regression or integer class identifiers for classification.

You must also define a function that implements the gini impurity score, as shown at Wikipedia:

```
def gini(y):
    "Return the gini impurity score for values in y"
```

Script `test_dtree_funcs.py` tests your implementation.

### Training algorithm

The training algorithm embodied by function `fit()` exhaustively tries combinations of features and feature values, looking for an optimal split.  The optimal split is one that splits a feature space for one feature into two sub-regions and the average variance (regression) or impurity (classification) is lower than that of the current node's observations.  

The first decision node is created by looking at the entire set of training records in X. Once split into two regions, training recursively splits those two regions. In this way, different subsamples of the training data are examined to create the decision nodes of the tree. If every decision node split the current set of samples exactly in half, than the height of the tree would be roughly `log(len(X))`.  Training returns a leaf node when there are less than or equal to `min_samples_leaf` observations in a subsample.

The algorithm looks like this:

```
fit(X, y, min_samples_leaf, loss):
    if X has fewer than min_samples_leaf observations, create and return a leaf node
    col, split = find_best_split(X, y, loss) # find best var and split value for X, y
    if col==-1, then we couldn't find a better split so return a leaf node
    split X into observations whose X[col] values are <= split and those > split
    recursively call fit() twice on these subsamples of X, y to get left and right children
    return a decision node with col, split, left child, and right child
```

Finding the optimal split looks like this:

```
find_best_split(X, y, loss):
    record loss(y) as the current best score (lowest variance or impurity)
    record variable -1 and split -1 as the best to initialize
    for each variable i:
        candidates = pick 11 values out of X[:, i] # (for speed reasons)
        for each split in candidates:
            left = y values <= split
            right = y values > split
            if left or right is empty then it's a bad split; try another candidate
            compute the loss for left and right chunks an average them
            if that combined loss is close to zero, we have perfection:
                return i, split
            if that combined loss is less than the best so far:
                track i, split, and that combined loss as the new best
    return the best i and split value
```

### Wrapping your functions in objects

After successfully building the functions that construct trees and make predictions, the next phase is to wrap or pull apart these functions and wrap them into class definitions: `RegressionTree621` and `ClassifierTree621` to mimic sklearn's `DecisionTreeClassifier` and `DecisionTreeRegressor` objects. Script `test_dtree.py` tests your implementation.

Object-oriented programming is probably unfamiliar to you, but there is plenty of material on the web (most of it is crap though).  You can check out [my OO summary](https://github.com/parrt/msds501/blob/master/notes/OO.ipynb), which sucks slightly less than other stuff on the web. 

The basic idea is that class definitions organize multiple functions together (functions within a class definition are called methods). For example, here is the skeleton class definitions that you will need:

```
class RegressionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini
    def fit(self, X, y):
        self.root = self.fit_(X, y)

    def fit_(self, X, y): # recursive version of fit()
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.
        """
        ...

    def predict(self, X_test):
        ...        
    def score(self, X_test, y_test):
        "See regressor_score() in test_dtree_funcs.py"
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

Download the [test scripts](https://github.com/parrt/msds621/tree/master/projects/dtree) and create blank script files `dtree_funcs.py` and `dtree.py` with perhaps `import numpy as np` as starter code.   I would focus on making sure that the functions work properly in `dtree_funcs.py` and and worry about the object-oriented implementation after all of your tests pass.

Cut/paste my definitions of `DecisionNode` and `LeafNode` into `dtree_funcs.py` if you plan on using those, but you are free to use your own binary tree implementation.

Define skeletons for functions `fit()`, `gini()`, and `predict()`.

Add the files to your repository, commit, and push back to github.

In this way, you have started on the project without actually having to do any work. Getting over inertia is an important step in any project.
 
## Deliverables

In your github repo `dtree`-*userid*, you must provide the following files at the root of the repository directory:

* `dtree_funcs.py` This is the initial implementation with the functions `fit()`, `gini()`, and `predict()` as well as the class definitions you need for decision tree implementation
* `dtree.py` This is the same code cut-and-paste into methods of class definitions to organize your code in an object-oriented way

I will copy in clean versions of the test scripts before grading your projects.

## Evaluation

Your code will be tested using the unit tests provided to you as part of this project. There are two regression and three classification toy data sets. Hopefully, getting even one of the tests to pass means you will get all of the test to pass. Nonetheless, each test is worth 17% for the function-based implementation that you start with. That means 85% of your grade  comes from getting the basic functionality to work.

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

Next, we will test the object-oriented version of your software using a similar script that simply invokes your objects as if they were sklearn objects. The functionality should not change in so you should get all of these test to pass if the test pass for the function-based code. With that in mind, there is an overall score of 15% given to you if you get the following tests to work; no partial credit for this part as they should all work or not work.

```
$ python -m pytest -v test_dtree.py 
============================================== test session starts ===============================================
platform darwin -- Python 3.7.1, pytest-4.0.2, py-1.7.0, pluggy-0.8.0 -- ...
cachedir: .pytest_cache
rootdir: /Users/parrt/courses/msds621-private/projects/dtree, inifile:
plugins: remotedata-0.3.1, openfiles-0.3.1, doctestplus-0.2.0, arraydiff-0.3
collected 5 items                                                                                                

test_dtree.py::test_boston PASSED                                                                          [ 20%]
test_dtree.py::test_california_housing PASSED                                                              [ 40%]
test_dtree.py::test_iris PASSED                                                                            [ 60%]
test_dtree.py::test_wine PASSED                                                                            [ 80%]
test_dtree.py::test_breast_cancer PASSED                                                                   [100%]

=========================================== 5 passed in 26.63 seconds ============================================
```

*My test scripts complete in less than 30 seconds and I will take off 10% if either of the test scripts take longer than one minute each. Please pay attention to efficiency.*