# Employee-Salary-Prediction-using-RandomForest-and-XGBoost

This project uses machine learning to predict employee salaries based on features like experience, education level, job title, and location. The final model is deployed as a web app using **Gradio** on Hugging Face Spaces, allowing users to interactively get salary estimates.

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Tech Stack](#tech-stack)
- [Models Used](#models-used)
- [Deployment](#deployment)
- [Visualizations](#visualizations)

---

## 🧠 Problem Statement

Estimating employee salaries accurately is essential for fair hiring practices and workforce planning. Traditional methods often lack consistency and transparency. This project addresses the issue by training machine learning models that predict salaries based on real-world input features. The interactive Gradio interface makes it easy to use by recruiters, HR teams, or job seekers.

---

## 🔧 Tech Stack

- **Language**: Python 3.10  
- **ML Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`  
- **Visualization**: `matplotlib`, `seaborn`  
- **Frontend**: Gradio  
- **Deployment**: Hugging Face Spaces (Gradio SDK)

---

## 📈 Models Used

| Model            | R² Score | MAE     |
|------------------|----------|---------|
| Random Forest    | ~0.83    | ~6000   |
| XGBoost Regressor| ~0.86    | ~5500   |

> 🔹 **Random Forest ** was chosen for final deployment due to better performance.

---

## 🚀 Deployment

🟢 **Live App**:  
[👉 Launch on Hugging Face](https://sahanapriyag-employee-salary-prediction.hf.space/?__theme=system&deep_link=Rg-lWUzrcgM)

📦 **Deployed using**: Hugging Face Spaces (Gradio SDK)

---

## 📊 Visualizations

- Correlation Heatmap
- Feature Importance (Random Forest & XGBoost)
- Predicted vs Actual Scatter Plot
- MAE and R² Comparison
- Residual and Distribution Plots

---
