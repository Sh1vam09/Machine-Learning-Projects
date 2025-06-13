#🔍 Credit Card Fraud Detection Using XGBoost

This project is a machine learning pipeline that detects fraudulent credit card transactions using the XGBoost Classifier. It tackles class imbalance with SMOTE and uses RandomizedSearchCV for hyperparameter tuning. The dataset used is the popular Kaggle Credit Card Fraud Dataset.

📁 Dataset
Source: Kaggle Credit Card Fraud Dataset
Size: 284,807 transactions
Features:
V1 to V28: PCA components
Amount, Time: Original features
Class: Target variable (0 → Non-Fraud, 1 → Fraud)

📊 Exploratory Data Analysis (EDA)
The EDA focuses on understanding class imbalance and identifying key features:

Visualized severe class imbalance (~0.17% fraud).

Plotted distribution of transaction Amount by class.

Heatmap of top 10 features most correlated with fraud.

⚙️ Preprocessing
Dropped the Time feature (not useful).

Scaled Amount using StandardScaler.

Handled class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

Split data using Stratified Train-Test Split to preserve fraud ratio.


🧪 Evaluation
Evaluated the final model on the unseen test set.
Metrics:
Classification Report (Precision, Recall, F1-Score)
ROC-AUC Score for probabilistic performance

Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn

📌 Project Highlights
End-to-end ML pipeline with visualization, preprocessing, tuning, and evaluation.

Tackles real-world issues like data imbalance and feature scaling.

Reproducible and scalable framework for binary classification problems.
