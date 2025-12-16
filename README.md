# 🏥 Insurance Charges Predictor: Robust ML Deployment

## 🎯 Project Overview

This project implements an end-to-end Machine Learning solution to predict individual annual medical insurance charges. The goal was to build a highly accurate and robust regression model, moving beyond basic Linear Regression to a tuned ensemble method, and deploying it into a functional web application.

The core objective was achieved by identifying and compensating for the high skewness in the target variable (`charges`) and leveraging highly influential features (like smoking status and BMI) to achieve a low Root Mean Squared Error (RMSE).

## 🚀 Deployment Status

The model is deployed locally using **Streamlit**, providing a clean web interface where users can input personal and health data to receive an instant prediction of their annual insurance cost.

## 📈 Model Performance Highlights

The project utilized a structured workflow to compare models, ultimately selecting a robust, optimized Random Forest Regressor.

| Metric | Linear Regression (Baseline) | **Random Forest Regressor (Optimized)** |
| :--- | :--- | :--- |
| **$R^2$ (Fit)** | 0.8041 | **0.8610** |
| **RMSE (Original $)** | $7,790.82 | **$4,284.74** |

The optimized model reduced the average prediction error by over **$3,500** compared to the baseline, achieving a final robust error of **$4,284.74**.

## 🔑 Key Feature Insights

Analysis of the optimized Random Forest model revealed the true drivers of insurance cost:

1.  **Smoker Status (`smoker_yes`):** ~50.7% Importance
2.  **Age (`age`):** ~39.5% Importance
3.  **Children (`children`):** ~3.4% Importance

These findings confirm that smoking status is the single most dominant factor in determining a person's insurance premium.

## 🛠️ Technology Stack

* **Language:** Python (3.x)
* **Core Libraries:** Pandas, NumPy, Matplotlib, Seaborn
* **Modeling:** Scikit-learn (Linear Regression, RandomForestRegressor, GridSearchCV)
* **Deployment:** Streamlit
* **Persistence:** Joblib

## 📁 Project Structure

The project follows a standard MLOps structure for clear separation of concerns:
