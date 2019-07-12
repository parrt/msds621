# Movie reviews sentiment analysis

Project 2, MS621<br>
*Fall 2019*

## Goal

In this project, you will build a multinomial naive bayes classifier to predict whether a movie review is positive or negative.  As part of the project, you will also learn to do *k*-fold cross validation testing.

You will do your work in a `bayes`-*userid* repository. Please keep all of your files in the root directory of the repository.

## Getting started

Download and uncompress [polarity data set v2.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz) into the root directory of your repository, but do not add the data to git.  My directory looks like:

```
$ ls
bayes.py   review_polarity  test_bayes.py
```

Download the [test_bayes.py](https://github.com/parrt/msds621/blob/master/projects/bayes/test_bayes.py) script into the root directory of your repository; you can add this if you want but I will overwrite it when testing. It assumes that the `review_polarity` directory is in the same directory (the root of the repository).

Download the [bayes.py starter kit](https://github.com/parrt/msds621/blob/master/projects/bayes/bayes.py) into the root directory of your repository. Make sure to add this to the repo.

See Naive Bayes discussion, p258 in [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/).

**Please do not add the data to your repository!**

## Discussion

A text classifier predicts to which class an unknown document belongs. In our case, the predictions are binary: 0 for negative movie review and 1 for positive movie review. Given document $d$ and a class $c$, we can think about classification mathematically as picking the most likely class:

$$
c^*= \frac{argmax}{c} P(c|d)
$$

Text classification requires a representation
 
+1 on top, |V| on bottom for smoothing. +1 on bottom because we’ve added one to V

Less powerful unless documents are very short than the word count isn't super meaningful and we should just go with the number of documents that have that word.

```
# Convert counts to {0,1}
# neg_binary = np.where(df_neg > 0, 1, 0)
neg_ndocs_with_word = neg_binary.sum(axis=0) # shape is (|V|,)
word_likelihood_neg = neg_ndocs_with_word / ndocs[0] # p(w,neg)
```
        
## Deliverables

To submit your project, ensure that your `bayes.py` file is submitted to your repository. That file must be in the root of your `bayes`-*userid* repository.  It should not have a main program and should consist of a collection of functions.

**Please do not add the data to your repository!**

## Submission

In your github `bayes`-*userid* repository, you should submit your `bayes.py` file in the root directory. It should not have a main program that runs when the file is imported.

*Please do not add data files to your repository!*

## Evaluation

To evaluate your projects I will run `test_bayes.py` from your repo root directory. Here is a sample test run:

```
$ python -m pytest -v test_bayes.py 
============================================== test session starts ============================
platform darwin -- Python 3.7.1, pytest-4.0.2, py-1.7.0, pluggy-0.8.0 -- ...
cachedir: .pytest_cache
rootdir: /Users/parrt/courses/msds621-private/projects/bayes, inifile:
plugins: remotedata-0.3.1, openfiles-0.3.1, doctestplus-0.2.0, arraydiff-0.3
collected 6 items                                                                                                

test_bayes.py::test_load PASSED                                                                            [ 16%]
test_bayes.py::test_vocab PASSED                                                                           [ 33%]
test_bayes.py::test_vectorize PASSED                                                                       [ 50%]
test_bayes.py::test_training_error PASSED                                                                  [ 66%]
test_bayes.py::test_kfold_621 PASSED                                                                       [ 83%]
test_bayes.py::test_kfold_sklearn_vs_621 PASSED                                                            [100%]

=========================================== 6 passed in 21.04 seconds ============================
```

Notice that it takes about 20 seconds. If your project takes more than one minute, I will take off 10 points from 100. Each test is evaluated in a binary fashion: it either works or it does not. Each failed test cost you 15 points.