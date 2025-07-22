import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load dataset from Hugging Face (or use a local .csv)
df = pd.read_csv("Salary Data.csv")
df = df.dropna()

# Train model
X = df.drop("Salary", axis=1)
y = df["Salary"]

categorical_cols = ["Gender", "Education Level", "Job Title"]
numerical_cols = ["Age", "Years of Experience"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)

# Gradio app
def predict_salary(age, gender, education, job_title, experience):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience
    }])
    prediction = model.predict(input_df)[0]
    return f"Predicted Salary: ${prediction:,.2f}"

demo = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Bachelor's", "Master's", "PhD"], label="Education Level"),
        gr.Textbox(label="Job Title"),
        gr.Number(label="Years of Experience")
    ],
    outputs="text",
    title="Employee Salary Predictor",
    description="Predict salary using a Random Forest model trained on employee data"
)

demo.launch()
