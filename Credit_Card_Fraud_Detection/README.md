# **🔍Credit Card Fraud Detection Using XGBoost**

This project is a machine learning pipeline that detects fraudulent credit card transactions using the XGBoost Classifier. It tackles class imbalance with SMOTE and uses RandomizedSearchCV for hyperparameter tuning. The dataset used is the popular Kaggle Credit Card Fraud Dataset.

## 📁 Dataset

Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Size: 284,807 transactions

Features:

V1 to V28: PCA components

Amount, Time: Original features

Class: Target variable (0 → Non-Fraud, 1 → Fraud)

## 📊 Exploratory Data Analysis (EDA)

The EDA focuses on understanding class imbalance and identifying key features:
Visualized severe class imbalance (~0.17% fraud).
Plotted distribution of transaction Amount by class.
Heatmap of top 10 features most correlated with fraud.

## ⚙️ Preprocessing

Dropped the Time feature (not useful).
Scaled Amount using StandardScaler.
Handled class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
Split data using Stratified Train-Test Split to preserve fraud ratio.


---

## 🧪 Evaluation

The model was evaluated on an unseen test set using key classification metrics:

| Metric        | Non-Fraud (0) | Fraud (1) |
| ------------- | ------------- | --------- |
| **Precision** | 1.00          | 0.45      |
| **Recall**    | 1.00          | 0.89      |
| **F1-Score**  | 1.00          | 0.60      |
| **Support**   | 56,864        | 98        |

* **Accuracy**: 1.00
* **Macro Avg F1-Score**: 0.80
* **Weighted Avg F1-Score**: 1.00
* **ROC-AUC Score**: **0.9785**

### 🔍 Interpretation:

* The model performs **perfectly on non-fraud cases**, which dominate the dataset.
* For the minority **fraud class**, it achieves:

  * **89% recall**: It detects most fraudulent transactions (which is critical).
  * **45% precision**: Some false positives, but acceptable tradeoff in fraud detection.
* **ROC-AUC Score of 0.9785** indicates **excellent class separation capability**.



## Requirements:
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
```  
## 📌 Project Highlights

End-to-end ML pipeline with visualization, preprocessing, tuning, and evaluation.
Tackles real-world issues like data imbalance and feature scaling.
Reproducible and scalable framework for binary classification problems.
