#!/usr/bin/env python
# coding: utf-8

# In[8]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib as jb

model = jb.load("kmeans_model.pkl")

app = FastAPI(title="Job Cluster Prediction API")



class JobData(BaseModel):

     age: int
     experience: float
     education: str
     job_role: str
     location: str
     gender: str
     industry: str
     company_size: str
     work_hours_per_week: float
     remote_work: str
     
@app.get("/")
def home():
    return {"message": "API Running"}


@app.post("/predict")
def predict(data:JobData):

    df = pd.DataFrame({

        'age': [data.age],
        'experience': [data.experience],
        'education': [data.education],
        'job_role': [data.job_role],
        'location': [data.location],
         'gender': [data.gender],
         'industry': [data.industry],
         'company_size': [data.company_size],
         'work_hours_per_week': [data.work_hours_per_week],
         'remote_work': [data.remote_work]

    })

    prediction = model.predict(df)

    return {"cluster": int(prediction[0])}

