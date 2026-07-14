import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# ==========================
# Load Model & Preprocessing
# ==========================

model = tf.keras.models.load_model("student_pass_model.keras")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

# ==========================
# Page Title
# ==========================

st.set_page_config(page_title="Student Pass Prediction", page_icon="🎓")

st.title("🎓 Student Pass Prediction")
st.write("Enter the student's information to predict whether they will Pass or Fail.")

# ==========================
# User Inputs
# ==========================

gender = st.selectbox(
    "Gender",
    encoders["gender"].classes_
)

race = st.selectbox(
    "Race/Ethnicity",
    encoders["race/ethnicity"].classes_
)

education = st.selectbox(
    "Parental Level of Education",
    encoders["parental level of education"].classes_
)

lunch = st.selectbox(
    "Lunch",
    encoders["lunch"].classes_
)

prep = st.selectbox(
    "Test Preparation Course",
    encoders["test preparation course"].classes_
)

reading_score = st.slider(
    "Reading Score",
    0,
    100,
    50
)

writing_score = st.slider(
    "Writing Score",
    0,
    100,
    50
)

# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    gender_enc = encoders["gender"].transform([gender])[0]
    race_enc = encoders["race/ethnicity"].transform([race])[0]
    education_enc = encoders["parental level of education"].transform([education])[0]
    lunch_enc = encoders["lunch"].transform([lunch])[0]
    prep_enc = encoders["test preparation course"].transform([prep])[0]

    data = np.array([[
        gender_enc,
        race_enc,
        education_enc,
        lunch_enc,
        prep_enc,
        reading_score,
        writing_score
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    probability = prediction[0][0]

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.success("✅ PASS")
    else:
        st.error("❌ FAIL")

    st.write(f"**Probability of Passing:** {probability:.2%}")