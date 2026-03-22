#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import joblib as jb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


df = pd.read_csv("job_salary_prediction_dataset.csv")

df.head()

X = df.drop(['salary'],axis =1)


if 'salary' in X.columns:
    X = X.drop('salary', axis=1)


numerical_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()


numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


model = Pipeline([
    ('pre', preprocessor),
    ('cluster', KMeans(n_clusters=3, random_state=42))
])

model.fit(X)


clusters = model.predict(X)

print(f'{clusters}')

jb.dump(model, "kmeans_model.pkl")

