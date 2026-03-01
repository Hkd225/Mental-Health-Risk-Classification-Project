# Mental Health Disorder Risk Assessment

## Project Overview

This project builds a Machine Learning system to predict an individual's mental health risk level 
based on demographic, lifestyle, and psychological factors.

The dataset is downloaded directly from Kaggle using kagglehub:

Dataset:
algozee/mental-health-disorder-risk-assessment-dataset

Target Variable:
mental_health_risk

Risk Categories:
0 → Low Risk
1 → Moderate Risk
2 → High Risk

---

## 1. Libraries Used

- pandas → Data manipulation
- scikit-learn → ML models and preprocessing
- xgboost → Gradient boosting classifier
- joblib → Model persistence
- kagglehub → Dataset download

---

## 2. Data Loading

Dataset is downloaded from Kaggle and loaded:

    path = kagglehub.dataset_download("algozee/mental-health-disorder-risk-assessment-dataset")
    df = pd.read_csv('mental_health_risk_dataset.csv')

Initial inspection:
- df.head()
- df.describe()
- df.info()

---

## 3. Feature and Target Separation

Features (X):
All columns except mental_health_risk

Target (y):
mental_health_risk

---

## 4. Train-Test Split

Dataset is split into:
- 80% Training
- 20% Testing

Using stratification to maintain class distribution:

    train_test_split(test_size=0.2, stratify=y, random_state=42)

---

## 5. Data Preprocessing

### 5.1 Label Encoding

Categorical columns encoded:
- gender
- marital_status
- education_level
- employment_status

Using LabelEncoder.

### 5.2 Feature Scaling

MinMaxScaler is applied to normalize all features into range [0,1].

---

## 6. Models Implemented

Four classification models are trained and compared:

1. Random Forest Classifier
2. Logistic Regression (L2 Regularization, Multinomial)
3. Support Vector Machine (RBF Kernel)
4. XGBoost Classifier

---

## 7. Model Training & Evaluation

Each model is evaluated using:

- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix (for Random Forest)

Example evaluation:

    accuracy_score(y_test, y_pred)
    classification_report(y_test, y_pred)

---

## 8. Model Comparison

Models Compared:
- Logistic Regression
- SVM
- XGBoost
- Random Forest

Performance is measured primarily by accuracy and classification metrics.

---

## 9. Model Saving (Deployment Ready)

All trained components are saved using joblib:

Saved files:
- model_rf.pkl
- model_logreg.pkl
- model_svm.pkl
- model_xgb.pkl
- scaler.pkl
- encoders.pkl

This enables future inference without retraining.

---

## 10. Inference with New Data

After reloading:

    loaded_model = joblib.load('model_logreg.pkl')

Steps for prediction:
1. Create input dictionary
2. Convert to DataFrame
3. Encode categorical features
4. Scale using saved scaler
5. Predict risk category
6. Map numeric result to risk label

Risk Mapping:

    0 → Low Risk (Risiko Rendah)
    1 → Moderate Risk (Risiko Sedang)
    2 → High Risk (Risiko Tinggi)

---

## 11. Batch Prediction Example

The notebook demonstrates prediction on:

- Single profile
- Five different profiles simultaneously

Additionally:
- predict_proba() is used to display model confidence percentage.

Example Output:

Profile 1:
- Low Risk: 85.2%
- Moderate Risk: 12.3%
- High Risk: 2.5%

---

## 12. System Workflow Summary

1. Download dataset
2. Data inspection
3. Feature-target split
4. Train-test split (stratified)
5. Encode categorical variables
6. Scale features
7. Train multiple ML models
8. Evaluate performance
9. Save models
10. Perform real-world prediction

---

## How to Run

1. Install dependencies:

    pip install pandas scikit-learn xgboost kagglehub joblib

2. Ensure Kaggle API access is configured.
3. Run notebook sequentially.
4. For testing new inputs, restart runtime and load saved models.

---

## Conclusion

This project demonstrates:

- End-to-end ML pipeline
- Multi-model comparison
- Feature preprocessing (encoding + scaling)
- Model persistence for deployment
- Probability-based prediction output

The system can be extended into:
- Web API (Flask / FastAPI)
- Streamlit dashboard
- Healthcare decision-support tool

---

Author: Muhammad Auffa Hakim Aditya
Mental Health Risk Classification Project
