# MSDS621 Introduction to Machine Learning

summary

# Administrivia

**INSTRUCTOR.** [Terence Parr](http://parrt.cs.usfca.edu). I’m a professor in the computer science and data science departments and was founding director of the MS in Analytics program at USF (which became the MS data science program).  Please call me Terence or Professor (the use of “Terry” is a capital offense).

**SPATIAL COORDINATES:**<br>

* Class is held at 101 Howard in Xth floor classroom Y.
* Exams are held in X and &. Both sections meet together.
* My office is room 607 @ 101 Howard up on mezzanine above the open area on 5th floor

**TEMPORAL COORDINATES.** Wed Oct X to Dec Y.

* Section 01: 
* Section 02: 
* Exams: 

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
|[Linear models](https://github.com/parrt/msds621/raw/master/projects/linreg/linreg.pdf)| 10%| XXX |
|[Naive Bayes](https://github.com/parrt/msds621/raw/master/projects/bayes/bayes.pdf) | 5% | XXX |
|[Decision trees]() | 13% | XXX |
|[Random Forest]() | 12% | XXX |
|Exam 1| 30%| 9AM-9:55AM XXX |
|Exam 2| 30%| 9AM-9:55AM XXX |

All projects will be graded with the specific input or tests given in the project description, so you understand precisely what is expected of your program. Consequently, projects will be graded in binary fashion: They either work or they do not. Each failed unit test gets a fixed amount off, no partial credit. The only exception is when your program does not run on the grader's or my machine because of some cross-platform issue. This is typically because a student has hardcoded some file name or directory into their program. In that case, we will take off just 10% instead of giving you a 0. Please go to github and verify that the website has the proper files for your solution. That is what I will download for testing.

Each project has a hard deadline and only those projects working correctly before the deadline get credit.  My grading script pulls from github at the deadline.  *All projects are due at the start of class on the day indicated, unless otherwise specified.*

**Grading standards**. I consider an **A** grade to be above and beyond what most students have achieved. A **B** grade is an average grade for a student or what you could call "competence" in a business setting. A **C** grade means that you either did not or could not put forth the effort to achieve competence. Below **C** implies you did very little work or had great difficulty with the class compared to other students.

# Syllabus

## Regularization for linear models

* [Review of linear models with scikit-learn](https://github.com/parrt/msds621/blob/master/notebooks/linear-models/sklearn-linear-models.ipynb)
* Demo need for regularization:
  * [Linear](https://github.com/parrt/msds621/blob/master/notebooks/linear-models/regressor-regularization.ipynb)
  * [Logistic](https://github.com/parrt/msds621/blob/master/notebooks/linear-models/classifier-regularization.ipynb)
* Linear regression L1, L2
* Logistic regression L1, L2
* [Project](https://github.com/parrt/msds621/raw/master/projects/linreg/linreg.pdf)

## Models

* kNN
* Naive Bayes ([Project](https://github.com/parrt/msds621/raw/master/projects/bayes/bayes.pdf))
* Decision Tree
* Random Forests

## Mechanics

* Data clean up
* Dealing with missing data

## Model assessment

* Loss functions, metrics
* ROC and PR curves

## Model interpretation

* Example-based model decision explanations
* Feature importance
  * Gini drop 
  * Permutation
  * Drop column
* Partial dependence

## Unsupervised learning

* k-means clustering
* Breiman's trick
