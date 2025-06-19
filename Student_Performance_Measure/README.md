# 🎓 Student Performance Prediction with Optuna

This project predicts students' exam scores based on academic and socioeconomic factors. It uses machine learning and hyperparameter optimization with **Optuna** to improve prediction accuracy.
## 📁 Dataset Overview

The dataset simulates real-world factors influencing student performance.

- 🔢 **Total Features Initially**: 20
- **Features**:
  - Hours_Studied
  - Attendance
  - Parental_Involvement
  - Access_to_Resources
  - Extracurricular_Activities
  - Sleep_Hours
  - Previous_Scores
  - Motivation_Level
  - Internet_Access
  - Tutoring_Sessions
  - Family_Income
  - Teacher_Quality
  - School_Type
  - Peer_Influence
  - Physical_Activity
  - Learning_Disabilities
  - Parental_Education_Level
  - Distance_from_Home
  - Gender
  - Exam_Score


- 🎯 **Target Variable**: `Exam_Score`


## 📌 Project Overview

- **Goal**: Predict `Exam_Score` of students using selected key features.
- **Tech Stack**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Optuna,XGBoost
- **Models Used**:
  - Support Vector Regression (SVR)
  - Random Forest Regressor
  - XGBoost Regressor
 
## 🧪 Feature Selection

- Started with **20 original features**.
- Used **Random Forest Regressor's feature importance** to identify most impactful features.
- Selected **important features**:
  - Access_to_Resources
  - Hours_Studied
  - Tutoring_Sessions
  - Attendance
  - Previous Scores
  - Parental_Education_Level

 
These features had the highest contribution to predicting students' performance.

## ⚙️ Workflow

1. **Data Preprocessing**
   - Cleaned dataset
   - Encoded categorical features
   - Scaled numerical values

2. **Feature Selection**
   - Applied `RandomForestRegressor().feature_importances_`
   - Selected top 7 features (see above)

3. **Model Training**
   - Compared regressors (SVR,XGBoost)

4. **Hyperparameter Tuning**
   - Performed using Optuna for automated tuning

5. **Evaluation**
   - Metrics: MAE, MSE, R² Score

## 🔍 Optuna Optimization

Optuna optimizes hyperparameters using efficient search algorithms.

Sample output:

Trial 0 finished with value: 0.5897634186767107 and parameters: {'classifier': 'XGBoost', 'xgb_n_estimators': 147, 'xgb_learning_rate': 0.1611238188497478, 'xgb_max_depth': 5, 'xgb_subsample': 0.729040018857276, 'xgb_colsample_bytree': 0.7999328277354354}. Best is trial 0 with value: 0.5897634186767107.

## 📈 Model Evaluation Metrics

After hyperparameter tuning and training, the best model achieved:

- **R² Score**: `0.7145`
- **Mean Squared Error (MSE)**: `4.0354`
- **Mean Absolute Error (MAE)**: `0.8959`

## 🧠 Sample Prediction

```python
test=df.sample(10).drop(['Sleep_Hours','Physical_Activity'],axis=1)
test_100= df[df['Exam_Score'] == 100].drop(['Sleep_Hours','Physical_Activity'],axis=1)
test=pd.concat([test,test_100])
ypred = model.predict(test)
```

| #   | Actual Score | Predicted Score       |
|-----|--------------|------------------------|
| 1   | 64           | 64.48                  |
| 2   | 64           | 64.44                  |
| 3   | 61           | 59.90                  |
| 4   | 68           | 66.52                  |
| 5   | 71           | 70.85                  |
| 6   | 67           | 68.44                  |
| 7   | 68           | 68.29                  |
| 8   | 67           | 67.91                  |
| 9   | 66           | 65.54                  |
| 10  | 61           | 62.76                  |
| 11  | 100          | 69.90                  |
| 12  | 100          | 73.85                  |

**Note** : The model performs well in the mid-range (60–80) where most of the data is concentrated — with **6414 values** in that range. It is not able to predict that good for extreme values.

🏁 Results Summary
R² Score: 0.7145 — indicates strong correlation between predicted and actual exam scores.

Mean Squared Error (MSE): 4.0354 — low error value suggests reliable predictions.

Mean Absolute Error (MAE): 0.8959 — on average, predictions deviate less than 1 mark.

Optuna Optimization: Helped identify the best SVM configuration (C, kernel, gamma) leading to improved accuracy.

Model Performance:

Performs very well in the mid-range scores (60–80), where most students lie (~6414 values).

Slightly underperforms on extreme values .

🔬 Best Model: SVM after Optuna tuning.
