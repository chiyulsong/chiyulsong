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


## 4. Evaludation

### Classification Evaluation Metrics
| **Metric**             | **Description**                                                  | **Use Case**                                     |
|-------------------------|------------------------------------------------------------------|-------------------------------------------------|
| **Accuracy**            | Proportion of correctly predicted instances over the total data | Suitable when classes are balanced              |
| **Precision**           | Proportion of positive predictions that are correct            | Important when reducing False Positives (e.g., spam filtering) |
| **Recall (Sensitivity)**| Proportion of actual positives correctly identified             | Important when reducing False Negatives (e.g., disease diagnosis) |
| **F1 Score**            | Harmonic mean of Precision and Recall                          | Useful when Precision and Recall are equally important |
| **ROC-AUC**             | Area under the ROC curve. Closer to 1 indicates better performance | Evaluates overall performance in binary classification |
| **Confusion Matrix**    | Table showing TP, TN, FP, and FN counts                        | Analyzes classification results in detail       |
| **Log Loss**            | Evaluates probabilistic predictions                            | Useful for probability-based models             |

### Regression Evaluation Metrics
| **Metric**                 | **Description**                                             | **Use Case**                                     |
|----------------------------|-----------------------------------------------------------|-------------------------------------------------|
| **Mean Absolute Error (MAE)** | Mean of the absolute differences between predictions and actual values | Intuitive evaluation of average error size      |
| **Mean Squared Error (MSE)**  | Mean of squared differences between predictions and actual values | Penalizes larger errors more heavily            |
| **Root Mean Squared Error (RMSE)** | Square root of MSE, interpretable in original units         | Similar to MSE but retains the original scale   |
| **R^2 Score**               | Proportion of variance explained by the model (closer to 1 is better) | Assesses overall explanatory power of the model |
| **Mean Absolute Percentage Error (MAPE)** | Mean of absolute percentage errors                        | Evaluates relative error                        |
| **Explained Variance Score** | Proportion of variance in the target explained by the model | Measures the model's explanatory capability     |




분류와 회귀 지표 선택 시 고려사항

- 데이터 특성:
불균형 데이터(예: 암 진단)에서는 Precision, Recall, F1 Score가 더 유용합니다.
균형 잡힌 데이터(예: 날씨 예측)에서는 Accuracy가 적합합니다.
- 비즈니스 목표:
분류 문제에서 False Positive와 False Negative의 비용 차이를 고려합니다.
회귀 문제에서는 MSE가 큰 오차를 더 강하게 페널티로 반영하기 때문에 중요한 경우 적합합니다.
- 모델 목적:
확률 기반 모델(예: Logistic Regression)에서는 Log Loss 또는 ROC-AUC가 더 중요합니다.
회귀에서 R^2은 전반적인 설명력을 평가하는 데 좋습니다.

추천
- 분류:
이진 분류 문제에서 F1 Score와 ROC-AUC를 조합해 평가.
다중 클래스 문제에서는 Confusion Matrix와 Precision-Recall 분석.
- 회귀:
MAE와 RMSE를 함께 사용하여 모델의 오차를 다각도로 평가.
R^2 점수를 추가하여 모델의 설명력을 확인.
지표에 대한 구체적인 사례나 코드 예제가 필요하면 말씀해주