# Comparative-Sentimental-Analysis-On-Movie-Reviews-Using-Machine-Learning-Algorithms

There are several steps taken to conduct this sentiment analysis, which is to collect data
using libraries in python, text processing, testing training data, and text classification using
logistic regression method.
From this classification or model it is possible to find whether the movie has either positive
or negative or neutral reviews of the movie. Here, the dataset is taken from the movie dataset,
which contains the review or the feedback of the movie. Then the combined data was tested
from the training data used for each presidential candidate to get an accuracy. In this paper,
the main focus is to anatomize the reviews conveyed by viewers on various movies and to
use this analysis to understand the customers’ sentiments and market behavior for better
customer experience. This research intends to analyze the reviews of customers on various
movies by implementing three algorithms namely K Nearest Neighbors, Logistic Regression
and Naive Bayes and provides conclusive remarks.
3.1 DATASET
The dataset can be obtained from the IMDB Dataset of 50K Movie Reviews. The dataset
contains 50 thousand movie reviews that have been pre-labeled with negative and positive
sentiment class labels.
3.2 DATA PRE-PROCESSING
Data preprocessing is the main step to prepare the data as of model requirements. The
preliminary step for any dataset is to be pre-processed while applying to any algorithm. This
is done by removing unwanted symbols, words or tags. These words or symbols do not affect
the result but consumes more time to be executed i.e may slow down the processing of an
algorithm. Our dataset involves the following steps
Cleaning the text
The text usually contains the html tags, which are considered as the unnecessary content ,
which do not have meaningful weight for analysing sentiment. So it is important to make
sure that this unnecessary content should be removed. In this phase the removal of noisy text
should be performed. The noisy data includes html tags, strips and brackets which doesn’t
manipulate the result. In this phase, the special characters are also removed.

25

Stemming
Stemming is the process of removing regular or frequent morphological endings from the
english words. This process stems the words to root words. For example, cool,cooler,coolest
are stemmed to its root word cool.
Remove stop words
This method removes the most frequently occurring or repeating words, these words are
considered as stop words which doesn’t have more use but consumes time, this reduces the
size of the dataset which is also known as stopping. As, an, the, is, etc are considered as stop
words.
3.3 ALGORITHMS

Logistic Regression
Logistic regression is basically a supervised classification algorithm. In a classification
problem, the target variable(or output), y, can take only discrete values for a given set of
features(or inputs), X.
Contrary to popular belief, logistic regression IS a regression model. The model builds a
regression model to predict the probability that a given data entry belongs to the category
numbered as “1”.

Algorithm
1. The logistic regression algorithm is imported from the scikit-learn package.
2. Split data into training and test data.
3. Generate a logistic regression model.
4. Train or fit the data into the model.
5. Predict the review

SVM Classifier
Support vector machines are a set of supervised learning methods used for classification,
regression, and outliers detection. A simple linear SVM classifier works by making a straight
line between two classes. That means all of the data points on one side of the line will

26

represent a category and the data points on the other side of the line will be put into a
different category. This means there can be an infinite number of lines to choose from.

Algorithm
We have two choices, we can either use the sci-kit learn library to import the SVM model
and use it directly or we can write our model from scratch.

1. It's really fun and interesting creating the model from scratch, but that requires a lot of
patience and time.
2. Instead, using a library from sklearn. SVM module which includes Support Vector
Machine algorithms will be much easier in implementation as well as to tune the
parameters.
Will be writing another article dedicated to the hand-on with the SVM algorithms
including classification and regression problems and we shall tweak and play tuning
parameters. Also will do a comparison on Kernel performance.
Multinomial Naive Bayesian
Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in
Natural Language Processing (NLP). The algorithm is based on the Bayes theorem and
predicts the tag of a text such as a piece of email or newspaper article. It calculates the
probability of each tag for a given sample and then gives the tag with the highest probability
as output.

Algorithm
1. Data Pre-processing step
2. Fitting Naive Bayes to the Training set
3. Predicting the test result
4. Test accuracy of the result (Creation of Confusion matrix)
5. Visualizing the test set result.

27

K Nearest Neighbours Classifier
K Nearest Neighbours classifier is perhaps the simplest and most widely used machine
learning algorithm. It can be applied to both classification problems and regression
problems. For smaller datasets, it outperforms most of the other classifiers.
KNN can be implemented by finding a group of k objects which are nearest to the test object,
and by assigning a label based on predominance of a class in the neighbourhood of the test
object.

Algorithm

1. The k-nearest neighbor algorithm is imported from the scikit-learn package.
2. Create feature and target variables.
3. Split data into training and test data.
4. Generate a k-NN model using neighbors value.
5. Train or fit the data into the model.
6. Predict the review.
