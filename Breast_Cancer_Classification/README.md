# 🔬 Breast Cancer Classification Using Logistic Regression

This project presents a complete machine learning pipeline to classify breast cancer tumors as **malignant** or **benign** using a **Logistic Regression** model. It includes preprocessing, hyperparameter tuning, cross-validation, evaluation, and performance visualization.

---

## 📁 Dataset

**Source:** Source: [Breast_Cancer_Data](https://github.com/Sh1vam09/Machine-Learning-Projects/blob/main/Breast_Cancer_Classification/breast_cancer.csv)
 
**Size:** 569 samples  
**Features:**
- 30 real-valued input features (e.g., radius, texture, perimeter, area, etc.)
- Target: `0` → Malignant, `1` → Benign

---

## 📊 Exploratory Data Analysis (EDA)

The EDA focused on understanding feature relationships and class distribution:

- Visualized the class balance (~38% malignant, ~62% benign)
- Plotted pairwise relationships and class-wise feature distributions
- Analyzed feature importance using correlation heatmaps

---

## ⚙️ Preprocessing

- **Standardization:** Applied `StandardScaler` to normalize all features.
- **Feature Selection:** Used all 30 features; no dimensionality reduction applied.
- **Data Splitting:** Stratified 80-20 train-test split to preserve class ratio.

---

## 🧠 Model & Hyperparameter Tuning

- **Model:** Logistic Regression (from `scikit-learn`)
- **Tuning:** Used `GridSearchCV` with 5-fold cross-validation for hyperparameter optimization
- **Cross-Validation Accuracy:** **97.80%**

---

## 🧪 Evaluation

The model was evaluated on the test set using classification metrics:

| Metric        | Malignant (0) | Benign (1) |
|---------------|---------------|------------|
| Precision     | 0.99          | 1.00       |
| Recall        | 1.00          | 0.98       |
| F1-Score      | 0.99          | 0.99       |
| Support       | 71            | 43         |

- **Accuracy:** **99.12%**
- **Macro Avg F1-Score:** 0.99
- **Weighted Avg F1-Score:** 0.99
- **ROC-AUC Score:** 0.995 (plotted for visual analysis)

---

## 📌 Interpretation

- The model generalizes well with **high precision and recall** for both classes.
- **Excellent recall (100%) for malignant tumors**, minimizing false negatives — critical in medical diagnosis.
- ROC-AUC score of **0.995** indicates near-perfect class separation.

---

## 🧰 Requirements

```bash
pandas  
numpy  
matplotlib  
seaborn  
scikit-learn  
