import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

@st.cache_resource
def load_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    pipe.fit(X, y)
    return pipe

model = load_model()

st.title("Diabetes Prediction App")

preg = st.number_input("Pregnancies", 0, 20, 2)
glucose = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 1000, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 35)

if st.button("Predict"):
    columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    df_input = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]], columns=columns)
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0]
    if pred == 1:
        st.error(f"Diabetes Risk Undi! Probability {prob[1]*100:.1f}%")
    else:
        st.success(f"No Diabetes Risk. Probability {prob[0]*100:.1f}%")