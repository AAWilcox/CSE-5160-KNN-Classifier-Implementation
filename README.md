<h1 align="center">KNN Classifier Implementation</h1>
<p align="center"><strong>CSE 5160 - Machine Learning</strong>
<br>By Alyssa Wilcox</p>
<br/>
<h2>About</h2>
The implementation of a Kth Nearest Neighbor (KNN) machine learning classifier. Using test data, this project classifies three varieties of wheat seeds: Kama, Rosa, and Canadian.

<h2>Course Information</h2>

- Student: Alyssa Wilcox
- Instructor: Dr. Yan Zhang
- Course: CSE 5160 Machine Learning
- Session: CSUSB Fall 2020

<h2>Training Set</h2>
This KNN clasifier heavily relies on training data to build the classifier.

Training Set Used:
- The training set used corresponds to 180 training instances of measurements and classification of wheat seeds.
- This data is provided by UCI's Machine Learning Respository.

Training Instance Features:
- Each training instance has seven features. Each measurement corresponds to some measurement taken from three varieties of wheat seeds.
- The seven features include:
  - Area, A
  - Perimeter, P
  - Compactness, C
  - Length of Kernel, L
  - Width of Kernel, W
  - Asymmetry Coefficient, AC
  - Length of Kernel Groove, LG

Training Instance Classifications:
- Each training instance is classified as one of three varieties of wheat seeds:
  1. Kama wheat seed, classified as 1
  2. Rosa wheat seed, classified as 2
  3. Canadian wheat seed, classified as 3

How the Training Set Built the Classifier:
- The classifier works by calculating the Euclidean distances between the test instance and every training instance.
- The classifications of the KNN training instances are used to classify a test instance.
- The training set does not explicitly build a model for the classifier.

<h2>Applying the Classifier on Test Instances</h2>
Test instances were used to determine the accuracy of the KNN classifier.

Test Set Data:
- The original data set consists of 210 training instances. Thirty of them are used as a test set, while the remaining 180 became the training set.
- The thirty test instances were chosen at random, with ten corresponding to Kama wheat seeds, ten corresponding to Rosa wheat seeds, and ten corresponding to Canadian wheat seeds.

Test Results - Test Error:
- To find test error, the KNN classifier was tested using a thirty instance long test set.
- Four different values of K were used: 5, 10, 15, and 20 nearest neighbors.
- Results:
  - K value 5:  Accuracy: 90%, Test Error: 10%
  - K value 10: Accuracy: 87%, Test Error: 13%
  - K value 15: Accuracy: 90%, Test Error: 10%
  - K value 20: Accuracy: 87%, Test Error: 13%

Test Results - Training Error:
- To find training error, the KNN classifier was tested using 10 randomly selected training instances.
- Four different values of K were used: 5, 10, 15, and 20 nearest neighbors.
- Results:
  - K value 5:  Accuracy: 100%, Training Error: 0%
  - K value 10: Accuracy: 100%, Training Error: 0%
  - K value 15: Accuracy: 100%, Training Error: 0%
  - K value 20: Accuracy: 100%, Training Error: 0%

<h2>Project status</h2>
Completed

<h2>Credits</h2>

- Training data provided by UCI's Machine Learning Respository: https://archive.ics.uci.edu/ml/datasets/seeds
- README.md template: https://gist.github.com/r4dixx/43e51e7d59027b26fefec2b389fc9e53#file-readme-student-md
