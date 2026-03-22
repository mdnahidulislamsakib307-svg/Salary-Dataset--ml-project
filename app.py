#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import requests

st.title("Job Cluster Prediction App")

age = st.number_input("Age", min_value=0)

experience = st.number_input("Experience (Years)", min_value=0.0)

education = st.selectbox(
    "Education",
    ["High School", "Bachelor", "Master", "PhD"]
)

job_role = st.text_input("Job Role")

location = st.text_input("Location")

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

industry = st.text_input("Industry")

company_size = st.selectbox(
    "Company Size",
    ["Small", "Medium", "Large"]
)

work_hours_per_week = st.number_input(
    "Work Hours Per Week",
    min_value=0.0
)

remote_work = st.selectbox(
    "Remote Work",
    ["Yes", "No"]
)


if st.button("Predict Cluster"):

    data = {

        "age": age,
        "experience": experience,
        "education": education,
        "job_role": job_role,
        "location": location,
        "gender": gender,
        "industry": industry,
        "company_size": company_size,
        "work_hours_per_week": work_hours_per_week,
        "remote_work": remote_work

    }

    response = requests.post(
        "https://salary-dataset-ml-project.onrender.com/predict",
        json=data
    )

    prediction = response.json()["prediction"]

    st.success(f"Predicted Cluster: {prediction}")

