# Machine Learning
Machine Learning is a subset of artificial intelligence that allows systems to learn and improve from experience without being explicitly programmed. It is categorized into three main types: Supervised, Unsupervised, and Reinforcement Learning.

## 📖 Table of Contents

1. [Machine Learning](#machine-learning)
    1. [Supervised Learning](#supervised-learning)
        1. [Regression](#regression)
            - [Linear Regression](#LR)
        2. [Classification](#classification)
            - [Logistic Regression](#LR)
            - [Decision Tree](#DT)
            - Random Forest
            - Support Vector Machine(SVM)
            - k-Nearest Neighbor(kNN)
            - Naive Bayes
        3. [Ensemble Methods](#ensemble-methods)
            - Bagging
            - Boosting
            - Voting
            - Stacking
    2. [Unsupervised Learning](#unsupervised-learning)
        1. [Clustering](#clustering)
            - k-means
            - DBSCAN
        2. [Dimensionality Reduction](#dimensionality-reduction)
            - PCA
        3. [Anomaly Detection](#anomaly-detection)
            - One-class SVM
            - Isolation Forest
        4. [Association Learning](#association-learning)
            - Apriori Algorithm
            - Eclat
    3. [Reinforcement Learning](#reinforcement-learning)

---



## 1. Supervised Learning
Supervised Learning uses labeled datasets to train algorithms to predict outcomes or classify data.

## 1.1 Regression
Regression is used to predict continuous values.

Linear Regression: Models the relationship between dependent and independent variables using a straight line.

## 1.2 Classification
Classification is used to predict discrete labels.

Logistic Regression: Predicts probabilities for binary outcomes.
Decision Tree: Uses a tree-like model of decisions for classification tasks.
Random Forest: Combines multiple decision trees for improved accuracy.
Support Vector Machine (SVM): Finds the optimal hyperplane for classification.
k-Nearest Neighbor (kNN): Classifies data points based on proximity to neighbors.
Naive Bayes: Applies Bayes' theorem assuming feature independence.

## 1.3 Ensemble Methods
Ensemble Methods combine multiple models to improve performance.

Bagging: Reduces variance by combining predictions of multiple models.
Boosting: Focuses on correcting errors made by prior models.
Voting: Aggregates predictions from multiple models to select the most frequent.
Stacking: Combines base models and uses another model to optimize predictions.


# 2. Unsupervised Learning
Unsupervised Learning finds hidden patterns or structures in unlabeled data.

## 2.1 Clustering
Clustering groups similar data points together.

k-means: Partitions data into k clusters by minimizing intra-cluster variance.
DBSCAN: Groups points based on density and detects outliers.

## 2.2 Dimensionality Reduction
Dimensionality Reduction reduces the number of features in data while retaining its core structure.

PCA (Principal Component Analysis): Projects data into lower dimensions by maximizing variance.

## 2.3 Anomaly Detection
Anomaly Detection identifies data points that deviate significantly from the norm.

One-class SVM: Identifies outliers by learning a boundary around normal data.
Isolation Forest: Detects anomalies by isolating data points in random partitions.


## 2.4 Association Learning
Association Learning discovers relationships between variables in datasets.

Apriori Algorithm: Finds frequent item sets and generates association rules.
Eclat: Uses intersection sets to find frequent item combinations.


# 3. Reinforcement Learning
Reinforcement Learning trains agents to make decisions by interacting with an environment to maximize rewards over time. Examples include training agents to play games or control robots.