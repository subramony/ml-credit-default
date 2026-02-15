🧠 Credit Card Default Prediction – Machine Learning Assignment

1️⃣ Problem Statement

The objective of this project is to build multiple classification models to predict whether a credit card client will default on their payment in the next month. The dataset contains demographic, financial, and repayment history information of customers.

The goal is to compare different machine learning models and evaluate their performance using multiple evaluation metrics.

2️⃣ Dataset Description

The dataset used is the Default of Credit Card Clients Dataset from Kaggle

Total Samples: 30,000

Total Features: 23

Target Variable: default.payment.next.month

Type: Binary Classification (0 = No Default, 1 = Default)

Key Features Include:

Credit limit (LIMIT_BAL)

Gender (SEX)

Education level (EDUCATION)

Marriage status (MARRIAGE)

Age

Past payment records (PAY_0 to PAY_6)

Bill amounts (BILL_AMT1–6)

Previous payments (PAY_AMT1–6)

The dataset shows moderate class imbalance (~78% non-default, ~22% default).

3️⃣ Models Implemented

The following six classification models were implemented:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors

Gaussian Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)

All models were evaluated using:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

4️⃣ Model Comparison Table

| Model               | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------- | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression | 0.808    | 0.708 | 0.688     | 0.241  | 0.357    | 0.326 |
| Decision Tree       | 0.723    | 0.609 | 0.382     | 0.405  | 0.393    | 0.214 |
| KNN                 | 0.794    | 0.694 | 0.553     | 0.346  | 0.426    | 0.320 |
| Naive Bayes         | 0.752    | 0.725 | 0.451     | 0.555  | 0.498    | 0.339 |
| Random Forest       | 0.814    | 0.755 | 0.637     | 0.366  | 0.465    | 0.382 |
| XGBoost             | 0.819    | 0.781 | 0.668     | 0.364  | 0.471    | 0.399 |

5️⃣ Observations

🔸 Logistic Regression

Logistic Regression achieved high precision but very low recall. It is conservative in predicting defaults and misses many defaulters. While overall accuracy was good, its ability to detect minority class instances was limited.

🔸 Decision Tree

The Decision Tree classifier showed improved recall compared to Logistic Regression but suffered from lower accuracy and AUC. The model likely overfitted the training data, resulting in weaker generalization performance.

🔸 K-Nearest Neighbors

KNN demonstrated balanced performance with moderate precision and recall. While it improved F1 score over Logistic Regression, it did not outperform ensemble methods in terms of AUC or MCC.

🔸 Naive Bayes

Naive Bayes achieved the highest recall and F1 score among all models. It effectively detected defaulters but at the cost of higher false positives. It is suitable when minimizing missed defaults is a priority.

🔸 Random Forest

Random Forest improved overall stability compared to a single Decision Tree. It achieved higher accuracy and AUC, indicating better generalization. The ensemble approach reduced overfitting.

🔸 XGBoost

XGBoost delivered the best overall performance with the highest accuracy, AUC, and MCC score. It provided strong class separation and balanced performance, making it the most effective model for this dataset.

🚀 How to Run the App Locally

python -m streamlit run app.py