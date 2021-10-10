# MSDS621 Introduction to Machine Learning

“*In God we trust; all others bring data.*” — Attributed to W. Edwards Deming and George Box

<img src="images/iris-TD-5-X-Arial.svg" width="300" align="right">This course introduces students to the key processes, models, and concepts of machine learning with a focus on:

* Regularization of linear models (finishing linear regression topic from previous class)
* Gradient descent loss minimization
* Naive bayes
* Nonparametric methods such as *k*-nearest neighbor
* Decision trees
* Random forests
* Mode interpretation
* Vanilla neural networks using pytorch

We study a few key models deeply, rather than providing a broad but superficial survey of models.  As part of the lab you will learn about data cleaning, feature engineering, and model assessment.

<img src="images/feynman.png" width="150" align="right" style="padding-top:10px">As part of this course, students implement linear and logistic regression with regularization through gradient descent, a Naive Bayes model for text sentiment analysis, decision trees, and random forest models. Implementing these models yourself is critical to truly understanding them. As Richard Feynman wrote, "**What I cannot create, I do not understand.**" (From his blackboard at the time of his death.) With an intuition behind how the models work, you'll be able to understand and predict their behavior much more easily.

# Class details

**INSTRUCTOR.** [Terence Parr](http://parrt.cs.usfca.edu). I’m a professor in the computer science and [data science program](https://www.usfca.edu/arts-sciences/graduate-programs/data-science) departments and was founding director of the MS in Analytics program at USF (which became the MS data science program).  Please call me Terence or Professor (“Terry” is not ok).

**SPATIAL COORDINATES**<br>

* Class is held at 101 Howard 5th floor classroom 529.
* Exams will be via HonorLock online/remote.
* My office is room 525 @ 101 Howard on 5th floor.

**TEMPORAL COORDINATES**<br>

Classes run Tue Oct 21 through Tue Dec 7. I believe we will have 12 class lectures, due to exams and Thanksgiving break.

* Lectures: 10AM-11:50AM (section 1) and 1-2:50PM (section 2)
* Exam 1: Thur Nov 11, 2021
* Exam 2: Tue Dec 7, 2021 (last day of class)

<!--
* Exams: Fri 5-6PM Nov 8; Fri 10-11:30AM Dec 6; Room 154-156
-->

**INSTRUCTION FORMAT**. Class runs for 1:50 hours, 2 days/week. Instructor-student interaction during lecture is encouraged and we'll mix in mini-exercises / labs during class. All programming will be done in the Python 3 programming language, unless otherwise specified.

**COURSE BOOK**. [The Mechanics of Machine Learning](https://mlbook.explained.ai/) (in progress)

**TARDINESS.** Please be on time for class. It is a big distraction if you come in late.

## Student evaluation

| Artifact | Grade Weight | Due date |
|--------|--------|--------|
|[Linear models](https://github.com/parrt/msds621/raw/master/projects/linreg/linreg.pdf)| 10%| Sun Oct 31, 11:59PM |
|[Naive Bayes](https://github.com/parrt/msds621/blob/master/projects/bayes/bayes.ipynb) | 8% | Tue Nov 9, 11:59PM (start of pm class) |
|[Decision trees](https://github.com/parrt/msds621/blob/master/projects/dtree/dtree.md) | 15% | Wed Nov 24, 11:59PM |
|[Random Forest](https://github.com/parrt/msds621/blob/master/projects/rf/rf.md) | 12% | Sun Dec 5, 11:59PM |
|Exam 1| 27%| Thu Nov 11|
|Exam 2| 28%| Tue, Dec 7|

All projects will be graded with the specific input or tests given in the project description, so you understand precisely what is expected of your program. Consequently, projects will be graded in binary fashion: They either work or they do not. Each failed unit test gets a fixed amount off, no partial credit. The only exception is when your program does not run on the grader's or my machine because of some cross-platform issue or some obviously trivial problem. (Attention to detail is critical.  For example, if you return an integer from a function and my code expects a string, your code makes my code fail.) This is typically because a student has hardcoded some file name or directory into their program. In that case, we will take off *a minimum* of 10% instead of giving you a 0, depending on the severity of the mistake.  Please go to github and verify that the website has the proper files for your solution. That is what I will download for testing.

For some projects, I run a small set of hidden tests that you do not have. These are typically worth 10% and the only way to get into the 90% range is to produce code that works on more than just the tests you're given.

**No partial credit**. Students are sometimes frustrated about not getting partial credit for solutions they labored on that do not actually work. Unfortunately, "almost working" just never counts in a job situation because nonfunctional solutions have no value.  We are not writing essays in English that have some value even if they are not superb.  When it comes to software, there is no fair way to assign such partial credit, other than a generic 30% or whatever for effort.  The only way to determine what is wrong with your project is for me to fix and/or complete the project. That is just not possible for 90 students. Even if that were possible, there is no way to fairly assign partial credit between students.  A few incorrect but critical characters can mean the difference between perfection and absolute failure. If it takes a student 20 hours to find that problem, is that worth more or less partial credit than another project that is half-complete but could be finished in five hours? To compensate, I try to test multiple pieces of the functionality in an effort to approximate partial credit.

Each project has a hard deadline and only those projects working correctly before the deadline get credit.  My grading script pulls from github at the deadline.  *All projects are due at the start of class on the day indicated, unless otherwise specified.*

*I reserve the right to change projects until the day they are assigned.*

**Grading standards**. I consider an **A** grade to be above and beyond what most students have achieved. A **B** grade is an average grade for a student or what you could call "competence" in a business setting. A **C** grade means that you either did not or could not put forth the effort to achieve competence. Below **C** implies you did very little work or had great difficulty with the class compared to other students.

# Syllabus

## Getting started

The first lecture is an overview of the entire machine learning process:

[Overview](https://github.com/parrt/msds621/raw/master/lectures/overview.pdf) (Day 1)

## Regularization for linear models

<img align="right" src="images/L1L2contour.png" width="180">

This topic more or less finishes off the linear regression course you just finished.

* [Review of linear models](https://github.com/parrt/msds621/raw/master/lectures/review-linear-models.pdf) (slides) (Day 1)
	* [Lab: Plotting decision surfaces for linear models](https://github.com/parrt/msds621/blob/master/labs/linear-models/decision-surfaces.ipynb) (Day 1)
* [Regularization of linear models L1, L2](https://github.com/parrt/msds621/raw/master/lectures/regularization.pdf) (slides) (Day 2)
	* [Lab: Exploring regularization for linear regression](https://github.com/parrt/msds621/blob/master/labs/linear-models/regularization-regr.ipynb) (Day 2)
	* [Lab: Regularization for logistic regression](https://github.com/parrt/msds621/blob/master/labs/linear-models/regularization-logi.ipynb) (Day 3)
	* See my deep dive: [A visual explanation for regularization of linear models](https://explained.ai/regularization/index.html)

## Training linear models with gradient descent

<img src="https://explained.ai/gradient-boosting/images/directions.png" width="120" align="right">This topic is required so we can train regularized linear models, and is critical to understanding neural networks that you'll study in a future class.

* [Gradient Descent optimization](https://github.com/parrt/msds621/raw/master/lectures/gradient-descent.pdf) (slides) (Day 3)
	* [Lab: Gradient descent in action](https://github.com/parrt/msds621/blob/master/labs/linear-models/gradient-descent.ipynb) (Day 3)
* (*[Regularization project](https://github.com/parrt/msds621/raw/master/projects/linreg/linreg.pdf)*)

## Models

<img align="right" src="images/boston-TD-3-X-Arial.png" width="300">

We will learn 3 models in depth for this course: naive bayes, decision trees, and random forests but will examine k-nearest-neighbor (kNN) briefly.

* [Naive Bayes](https://github.com/parrt/msds621/raw/master/lectures/naive-bayes.pdf)  (slides) (Day 4)
  * [Lab: Naive bayes by hand](https://github.com/parrt/msds621/blob/master/labs/bayes/naive-bayes.ipynb) (Day 4)
  * (*[Naive Bayes project](https://github.com/parrt/msds621/blob/master/projects/bayes/bayes.ipynb)*)
* [Intro to non-parametric machine learning models](https://github.com/parrt/msds621/raw/master/lectures/nonparametric-models.pdf) (slides) (Day 5)
* [Decision trees](https://github.com/parrt/msds621/raw/master/lectures/decision-trees.pdf) (slides) (Day 5)
  * [Lab: Partitioning feature space](https://github.com/parrt/msds621/blob/master/labs/trees/partitioning-feature-space.ipynb) (Day 6)
  * [Binary tree crash course](https://github.com/parrt/msds621/raw/master/lectures/binary-trees.pdf) (slides) (Day 6)
  * [Lab: Binary trees](https://github.com/parrt/msds621/blob/master/labs/trees/binary-trees.ipynb) (Day 6)
  * [Training decision trees](https://github.com/parrt/msds621/raw/master/lectures/training-decision-trees.pdf) (slides) (Day 7)
  * (*[Decision trees project](https://github.com/parrt/msds621/blob/master/projects/dtree/dtree.md)*) 
* [Random Forests](https://github.com/parrt/msds621/raw/master/lectures/random-forests.pdf) (slides) (Day 7, Day 8)
  * [Lab: Exploring Random Forests](https://github.com/parrt/msds621/blob/master/labs/trees/random-forests.ipynb) (Day 8)
  * [Out-of-bag (OOB) validation sets](https://github.com/parrt/msds621/raw/master/lectures/rf-oob.pdf) (slides) (Day 8)
  * (*[Random Forest project](https://github.com/parrt/msds621/blob/master/projects/rf/rf.md)*)

## Model interpretation

<img src="images/heatmap.png" width="150" align="right">

* [Feature importance](https://github.com/parrt/msds621/raw/master/lectures/feature-importance.pdf) (slides) (Day 9)
	* Gini-drop importance for random forests
	* Drop-column importance
	* Permutation importance
	* Null-distribution importance
* Partial dependence

## Unsupervised learning

<img src="images/clustering.png" width="150" align="right">

Clustering isn't used nearly as much as supervised learning, but it's an important part of your education  and is extremely useful in in certain circumstances, such as image color quantization. (Image credit [Wikipedia](https://en.wikipedia.org/wiki/Color_quantization).)

* [Clustering](https://github.com/parrt/msds621/raw/master/lectures/clustering.pdf) (slides)  (Day 9, Day 10)
  * k-means clustering
  * Hierarchical clustering
  * Breiman's trick for clustering with RFs

## Vanilla deep learning networks (Day 10-12)

* [Fundamentals of deep learning regressors and classifiers](https://github.com/parrt/msds621/raw/master/lectures/deep-learning.pdf) (slides)
* [intro-regression-training-cars.ipynb](notebooks/deep-learning/1.intro-regression-training-cars.ipynb)<br>Load toy cars data set and train regression models to predict miles per gallon (MPG) through a variety of techniques. We start out doing a brute force grid search of many different slope and intercept (m, b) model parameters, looking for the best fit. Then we manually compute partial derivatives of the loss function and perform gradient descent using plain numpy. We look at the effect on the loss function of normalizing numeric variables to have zero mean and standard deviation one. Finally, this notebook shows you how to use the autograd (auto differentiation) functionality of pytorch as a way to transition from numpy to pytorch training loops.
* [pytorch-nn-training-cars.ipynb](notebooks/deep-learning/2.pytorch-nn-training-cars.ipynb)<br>Once we can implement our own gradient descent using pytorch autograd and matrix algebra, it's time to graduate to using pytorch's built-in neural network module and the built-in optimizers (e.g., Adam). Next, we observe how a sequence of two linear models is effectively the same as a single linear model. After we add a nonlinearity, we see more sophisticated curve fitting. Then we see how a sequence of multiple linear units plus nonlinearities affects predictions. Finally, we see what happens if we give a model too much power: the regression curve over fits the training data.
* [train-test-diabetes.ipynb](notebooks/deep-learning/3.train-test-diabetes.ipynb)<br>This notebook explores how to use a validation set to estimate how well a model generalizes from its training data to unknown test vectors. We will see that deep learning models often have so many parameters that we can drive training loss to zero, but unfortunately the validation loss grows as the model overfits. We will also compare how deep learning does compared to a random forest model as a baseline.
* [binary-classifier-wine.ipynb](notebooks/deep-learning/4.binary-classifier-wine.ipynb)<br>Shifting to binary classification now, we consider the toy wine data set and build models that use features proline and alcohol to predict wine classification (class 0 or class 1). We will add a sigmoid activation function to the final linear layer, which will give us the probability that an input vector represents class 1. A single linear layer plus the sigmoid yields a standard logistic regression model. By adding another linear layer and nonlinearity, we see a curved decision boundary between classes. By adding lots of neurons and more layers, we see even more complex decision boundaries appear.
* [multiclass-classifier-mnist.ipynb](notebooks/deep-learning/5.multiclass-classifier-mnist.ipynb)<br>To demonstrate k class classification instead of binary classification, we use the traditional MNIST digital image recognition problem. We'll again use a random forest model as a baseline classifier. Instead of a sigmoid on a single output neuron, k class classifiers use k neurons in the final layer and then a softmax computation instead of a simple sigmoid. We see fairly decent recognition results with just 50 neurons.  By using 500 neurons, we get slightly better results.
* [gpu-mnist.ipynb](notebooks/deep-learning/6.gpu-mnist.ipynb)<br>This notebook redoes the examples from the previous MNIST notebook but using the GPU to perform matrix algebra in parallel. We use `.to(device)` on tensors and models to shift them to the memory on the GPU. The model trains much faster using the huge number of processors on the GPU. You will need to <a href="https://colab.research.google.com/github/parrt/fundamentals-of-deep-learning/blob/main/notebooks/7.gpu-mnist.ipynb">run the notebook at colab</a> or from an AWS machine to get access to a GPU.
* [SGD-minibatch-mnist.ipynb](notebooks/8.SGD-minibatch-mnist.ipynb)<br>We have been doing batch gradient descent, meaning that we compute the loss on the complete training set as a means to update the parameters of the model. If we process the training data in chunks rather than a single batch, we call it mini-batch gradient descent, or more commonly stochastic gradient descent (SGD). It is called stochastic because of the imprecision and, hence, randomness introduced by the computation of gradients on a subset of the training data. We tend to get better generalization with SGD; i.e., smaller validation loss.

## Supporting resources

A lot of the mechanics of machine learning will be covered in the machine learning lab, but here are some notebooks and slides that could be of use to you.

### Notebooks

There are a number of notebooks associated with this course that could prove useful to you:

* [Linear models, regularization](https://github.com/parrt/msds621/tree/master/notebooks/linear-models/)
* [Binary trees, decision trees, bias-variance](https://github.com/parrt/msds621/tree/master/notebooks/trees/)
* [Model assessment](https://github.com/parrt/msds621/tree/master/notebooks/assessment/)
* [Clustering](https://github.com/parrt/msds621/tree/master/notebooks/clustering/)

The following notebook takes you through a number of important processes, which you are free to do at your leisure. Even if we haven't covered the topics in lecture, you can still get something out of the notebook.

* [Getting a sense of the training and testing procedure notebook](https://github.com/parrt/msds621/blob/master/notebooks/process/basic-process.ipynb)

### Slides

#### Model assessment

<img src="images/log-dec.svg" width="150" align="right">

* [Bias-variance trade-off](https://github.com/parrt/msds621/raw/master/lectures/bias-variance.pdf) (slides)
* [Model assessment](https://github.com/parrt/msds621/raw/master/lectures/model-assessment.pdf) (slides)
* [Regressor and classifier metrics](https://github.com/parrt/msds621/raw/master/lectures/metrics.pdf) (slides)

#### Mechanics

<img src="images/split-str.png" width="150" align="right">

* [Preparing data for modeling](https://github.com/parrt/msds621/raw/master/lectures/data-prep.pdf) (slides)
* [Basic feature engineering](https://github.com/parrt/msds621/raw/master/lectures/feature-engineering.pdf)



# Administrivia

**HONORLOCK**. All tests use **HonorLock** via Canvas and have strict time limits. You will be unable to do anything other than take the test; no access to the Internet etc.  A proctor will monitor you during exams to ensure you do not communicate with anyone else during the test. Generally speaking, HonorLock will record all your web, computer, and personal activities (e.g., looking at your phone) during the quiz. It will flag suspicious behavior for my review and will save the recordings for 6 months if I need to go back and check it.

Please see the [How to use" page for students](https://honorlock.kb.help/-students-starting-exam/how-to-use-honorlock-student/). Either I or another instructor will launch a practice quiz on Canvas during the first week of class to ensure everything is set up properly.

* Google Chrome and a webcam are required. At the beginning of the quiz, you will be able to add the Chrome extension for Honorlock, then follow the instructions to share your screen and record your quiz.
* You might be asked to change settings on your computer while doing this. You can change the setting and come back to the quiz. This change should only be expected once.
* If you are showing us the side view of your face we don’t know if you’ve got an earbud in your other ear. This is not allowed.  
* Make sure you are facing into the camera as Honorlock will shut down the system and force you to restart.
* Make sure that you are not looking down and to the right as if you are looking at notes or using your phone. Honorlock will flag this as cheating.
* You must not start and stop your browser; Honorlock will flag this is cheating.
* You must not use other applications or visit non-Canvas-quiz URLs during the exam unless the exam indicates this is permitted.
* Do not have your phone visible as the proctor will stop the quiz

Side notes:

* Start the quiz with a single Chrome window and single tab in that window.
* When the "share screen button" is grey, you can still click it and it will work.
* HonorLock flags activities other than the allowed ones: for example when you are accessing a website other than canvas or looking at your phone. I will evaluate these cases and make a judgment myself. I will reach out to you when necessary. If you have followed the guidelines, you don’t need to worry.
* If you have an honorlock software issue during the test, you must take a screen picture with your phone or ipad and notify me immediately via private slack to timestamp the situation with the picture and reason why you cannot proceed. Please contact tech support on the screen to resolve (they are very quick). I will check the Honorlock recording and timestamp of your pictures to grade. 
* [Privacy statement from HonorLock](https://honorlock.com/student-privacy-statement/) just in case you are worried about privacy. Since access to Honorlock is very limited, and you are expected to only work on the quiz during the proctoring time, the data that Honorlock records is very limited too. The data storage and sharing agreement don’t have a higher risk than your regular school actives (Zoom, email, Canvas, ...). 

**ACADEMIC HONESTY.** You must abide by the copyright laws of the United States and academic honesty policies of USF. You may not copy code from other current or previous students. All suspicious activity will be investigated and, if warranted, passed to the Dean of Sciences for action.  Copying answers or code from other students or sources during a quiz, exam, or for a project is a violation of the university’s honor code and will be treated as such. Plagiarism consists of copying material from any source and passing off that material as your own original work. Plagiarism is plagiarism: it does not matter if the source being copied is on the Internet, from a book or textbook, or from quizzes or problem sets written up by other students. Giving code or showing code to another student is also considered a violation.

The golden rule: **You must never represent another person’s work as your own.**

If you ever have questions about what constitutes plagiarism, cheating, or academic dishonesty in my course, please feel free to ask me.

**Note:** Leaving your laptop unattended is a common means for another student to take your work. It is your responsibility to guard your work. Do not leave your printouts laying around or in the trash. *All persons with common code are likely to be considered at fault.*

**USF policies and legal declarations**

*Students with Disabilities*

If you are a student with a disability or disabling condition, or if you think you may have a disability, please contact <a href="/sds">USF Student Disability Services</a> (SDS) for information about accommodations.

*Behavioral Expectations*

All students are expected to behave in accordance with the <a href="https://usfca.edu/fogcutter">Student Conduct Code</a> and other University policies.

*Academic Integrity*

USF upholds the standards of honesty and integrity from all members of the academic community. All students are expected to know and adhere to the University's <a href="https://usfca.edu/academic-integrity/">Honor Code</a>.

*Counseling and Psychological Services (CAPS)*

CAPS provides confidential, free <a href="https://usfca.edu/student-health-safety/caps">counseling</a> to student members of our community.

*Confidentiality, Mandatory Reporting, and Sexual Assault*

For information and resources regarding sexual misconduct or assault visit the <a href="https://myusf.usfca.edu/title-ix">Title IX</a> coordinator or USFs <a href="http://usfca.callistocampus.org" target="_blank">Callisto website</a>.

