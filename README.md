# MSDS621 Introduction to Machine Learning

“*In God we trust; all others bring data.*” — Attributed to W. Edwards Deming and George Box


This course introduces students to the key processes and concepts of machine learning (for table-like structured data), such as data cleaning, dealing with missing data, basic feature engineering, model training, feature selection, and model assessment. We study a few key models deeply, rather than providing a broad but superficial survey of models. <img src="images/feynman.png" width="200" align="right">As part of this course, students implement linear and logistic regression with regularization through gradient descent, a Naive Bayes model for text sentiment analysis, decision trees, and random forest models.   Implementing these models yourself is critical to truly understanding them. As Richard Feynman wrote, "*What I cannot create, I do not understand.*" (From his blackboard at the time of his death.)

# Administrivia

**INSTRUCTOR.** [Terence Parr](http://parrt.cs.usfca.edu). I’m a professor in the computer science and [data science program](https://www.usfca.edu/arts-sciences/graduate-programs/data-science) departments and was founding director of the MS in Analytics program at USF (which became the MS data science program).  Please call me Terence or Professor (the use of “Terry” is a capital offense).

**SPATIAL COORDINATES:**<br>

* Class is held at 101 Howard in Xth floor classroom Y.
* Exams are held in X and &. Both sections meet together.
* My office is room 607 @ 101 Howard up on mezzanine above the open area on 5th floor

**TEMPORAL COORDINATES.** Thu Oct 17 to Tue Dec 3.

* Section 01: 10 - 11:50 AM Room 155-156
* Section 02: 1:15 - 3:05 PM Room 155-156
* Exams: Nov 8, Dec 6

**INSTRUCTION FORMAT**. Class runs for 1:50 hours, 2 days/week. Instructor-student interaction during lecture is encouraged and we'll mix in mini-exercises / labs during class. All programming will be done in the Python 3 programming language, unless otherwise specified.

**COURSE BOOK**. [The Mechanics of Machine Learning](https://mlbook.explained.ai/) (in progress)

**TARDINESS.** Please be on time for class. It is a big distraction if you come in late.

**ACADEMIC HONESTY.** You must abide by the copyright laws of the United States and academic honesty policies of USF. You may not copy code from other current or previous students. All suspicious activity will be investigated and, if warranted, passed to the Dean of Sciences for action.  Copying answers or code from other students or sources during a quiz, exam, or for a project is a violation of the university’s honor code and will be treated as such. Plagiarism consists of copying material from any source and passing off that material as your own original work. Plagiarism is plagiarism: it does not matter if the source being copied is on the Internet, from a book or textbook, or from quizzes or problem sets written up by other students. Giving code or showing code to another student is also considered a violation.

The golden rule: **You must never represent another person’s work as your own.**

If you ever have questions about what constitutes plagiarism, cheating, or academic dishonesty in my course, please feel free to ask me.

**Note:** Leaving your laptop unattended is a common means for another student to take your work. It is your responsibility to guard your work. Do not leave your printouts laying around or in the trash. *All persons with common code are likely to be considered at fault.*

**USF policies and legal declarations**

*Students with Disabilities*

If you are a student with a disability or disabling condition, or if you think you may have a disability, please contact <a href="/sds">USF Student Disability Services</a> (SDS) for information about accommodations.

*Behavioral Expectations*

All students are expected to behave in accordance with the <a href="/fogcutter">Student Conduct Code</a> and other University policies.

*Academic Integrity*

USF upholds the standards of honesty and integrity from all members of the academic community. All students are expected to know and adhere to the University's <a href="/academic-integrity/">Honor Code</a>.

*Counseling and Psychological Services (CAPS)*

CAPS provides confidential, free <a href="/student-health-safety/caps">counseling</a> to student members of our community.

*Confidentiality, Mandatory Reporting, and Sexual Assault*

For information and resources regarding sexual misconduct or assault visit the <a href="/TITLE-IX">Title IX</a> coordinator or USFs <a href="http://usfca.callistocampus.org" target="_blank">Callisto website</a>.

## Student evaluation

| Artifact | Grade Weight | Due date |
|--------|--------|--------|
|[Linear models](https://github.com/parrt/msds621/raw/master/projects/linreg/linreg.pdf)| 10%| Mon Oct 31, 11:59PM |
|[Naive Bayes](https://github.com/parrt/msds621/raw/master/projects/bayes/bayes.pdf) | 8% | Monday Nov 11, 11:59PM |
|[Decision trees](https://github.com/parrt/msds621/blob/master/projects/dtree/dtree.md) | 15% | Mon Nov 21, 11:59PM |
|[Random Forest](https://github.com/parrt/msds621/blob/master/projects/rf/rf.md) | 12% | Sun Dec 8, 11:59PM |
|Exam 1| 25%| Fri Nov 8, 5-6:15PM |
|Exam 2| 30%| Fri, Dec 11 10AM-11:30AM |

All projects will be graded with the specific input or tests given in the project description, so you understand precisely what is expected of your program. Consequently, projects will be graded in binary fashion: They either work or they do not. Each failed unit test gets a fixed amount off, no partial credit. The only exception is when your program does not run on the grader's or my machine because of some cross-platform issue. This is typically because a student has hardcoded some file name or directory into their program. In that case, we will take off *a minimum* of 10% instead of giving you a 0, depending on the severity of the mistake.  Please go to github and verify that the website has the proper files for your solution. That is what I will download for testing.

Each project has a hard deadline and only those projects working correctly before the deadline get credit.  My grading script pulls from github at the deadline.  *All projects are due at the start of class on the day indicated, unless otherwise specified.*

**Grading standards**. I consider an **A** grade to be above and beyond what most students have achieved. A **B** grade is an average grade for a student or what you could call "competence" in a business setting. A **C** grade means that you either did not or could not put forth the effort to achieve competence. Below **C** implies you did very little work or had great difficulty with the class compared to other students.

# Syllabus

[Overview](https://github.com/parrt/msds621/raw/master/lectures/intro.pdf)
 
## Regularization for linear models

* [Review of linear models](https://github.com/parrt/msds621/raw/master/lectures/review-linear-models.pdf) (slides)
* Notebooks demonstrating need for regularization:
  * [Linear models with scikit-learn notebook](https://github.com/parrt/msds621/blob/master/notebooks/linear-models/sklearn-linear-models.ipynb)
  * [Linear regularization notebook](https://github.com/parrt/msds621/blob/master/notebooks/linear-models/regressor-regularization.ipynb)
  * [Logistic regularization notebook](https://github.com/parrt/msds621/blob/master/notebooks/linear-models/classifier-regularization.ipynb)
* [Regularization of linear models L1, L2](https://github.com/parrt/msds621/raw/master/lectures/regularization.pdf) (slides)
* [Gradient Descent optimization](https://github.com/parrt/msds621/raw/master/lectures/gradient-descent.pdf) (slides)
* (*[Regularization project](https://github.com/parrt/msds621/raw/master/projects/linreg/linreg.pdf)*)

## Models

We will learn 3 models in depth for this course: naive bayes, decision trees, and random forests but will examine k-nearest-neighbor (kNN) briefly.

* [Naive Bayes](https://github.com/parrt/msds621/raw/master/lectures/naive-bayes.pdf)  (slides)
  * (*[Naive Bayes project](https://github.com/parrt/msds621/raw/master/projects/bayes/bayes.pdf)*)
* [Intro to non-parametric machine learning models](https://github.com/parrt/msds621/raw/master/lectures/nonparametric-models.pdf) (slides)
* [Decision trees](https://github.com/parrt/msds621/raw/master/lectures/decision-trees.pdf) (slides)
  * [Binary tree crash course](https://github.com/parrt/msds621/raw/master/lectures/binary-trees.pdf) (slides)
  * [Training decision trees](https://github.com/parrt/msds621/raw/master/lectures/training-decision-trees.pdf) (slides)
  * [Decision trees notebook](https://github.com/parrt/msds621/blob/master/notebooks/trees/partitioning.ipynb)
  * [Binary trees notebook](https://github.com/parrt/msds621/blob/master/notebooks/trees/basics.ipynb)
  * (*[Decision trees project](https://github.com/parrt/msds621/blob/master/projects/dtree/dtree.md)*)
* [Random Forests](https://github.com/parrt/msds621/raw/master/lectures/random-forests.pdf) (slides)
  * (*[Random Forest project](https://github.com/parrt/msds621/blob/master/projects/rf/rf.md)*)

## Mechanics

* [Getting a sense of the training and testing procedure notebook](https://github.com/parrt/msds621/blob/master/notebooks/process/basic-process.ipynb)
* [Preparing data for modeling](https://github.com/parrt/msds621/raw/master/lectures/data-prep.pdf) (slides)
* [Basic feature engineering](https://github.com/parrt/msds621/raw/master/lectures/feature-engineering.pdf) (slides)

## Model assessment

* [Bias-variance trade-off](https://github.com/parrt/msds621/raw/master/lectures/bias-variance.pdf) (slides)
* [Model assessment](https://github.com/parrt/msds621/raw/master/lectures/model-assessment.pdf) (slides)

## Model interpretation

* [Feature importance](https://github.com/parrt/msds621/raw/master/lectures/feature-importance.pdf) (slides)
* Partial dependence

## Unsupervised learning

* [Clustering](https://github.com/parrt/msds621/raw/master/lectures/clustering.pdf) (slides)
  * k-means clustering
  * Hierarchical clustering
  * Breiman's trick for clustering with RFs