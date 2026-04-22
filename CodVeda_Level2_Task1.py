#!/usr/bin/env python
# coding: utf-8

# # Customer Retention Intelligence System
# 
# ## Problem Statement
# Predict whether a telecom customer is likely to churn using behavioral and usage-based features.
# 
# ---
# 
# ## What makes this project UNIQUE?
# Unlike basic churn models, this notebook includes:
# - Smart feature engineering (usage intensity, charge ratios)
# - Class imbalance handling
# - Probability threshold tuning (business-focused decision making)
# - Interpretable Logistic Regression with insights
# 
# ---
# 
# ## Tech Stack
# - Python (Pandas, NumPy)
# - Scikit-learn (ML + preprocessing)
# - Matplotlib & Seaborn (visualization)
# 
# ---
# 
# ## Business Impact
# Helps telecom companies:
# - Identify high-risk customers early
# - Optimize retention campaigns
# - Reduce revenue loss

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve


# ## Data Loading & Merging

# In[2]:


train = pd.read_csv("churn-bigml-80.csv")
test = pd.read_csv("churn-bigml-20.csv")

df = pd.concat([train, test], ignore_index=True)
df.head()


# ## Initial Data Exploration

# In[3]:


print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Target distribution
sns.countplot(x=df["Churn"])
plt.title("Churn Distribution")
plt.show()


# ## Data Cleaning + Feature Engineering (UNIQUE PART)

# In[4]:


# Drop irrelevant column
if 'State' in df.columns:
    df.drop('State', axis=1, inplace=True)

# Convert categorical columns
df['International plan'] = df['International plan'].map({'Yes':1, 'No':0})
df['Voice mail plan'] = df['Voice mail plan'].map({'Yes':1, 'No':0})

# Feature Engineering (UNIQUE)
df['Total_calls'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls']
df['Total_minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes']

df['Charge_per_min'] = df['Total day charge'] / (df['Total day minutes'] + 1)

df.head()


# ## Feature & Target Separation

# In[5]:


X = df.drop('Churn', axis=1)
y = df['Churn']


# ## Train-Test Split

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ## Feature Scaling

# In[7]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Handling Class Imbalance (IMPORTANT)

# In[8]:


model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)


# ## Model Evaluation

# In[9]:


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))


# ## ROC-AUC Score

# In[10]:


y_prob = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", auc)

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# ## Threshold Optimization (VERY UNIQUE)

# In[11]:


threshold = 0.3  # tuned for better recall

y_pred_custom = (y_prob > threshold).astype(int)

print("New Accuracy:", accuracy_score(y_test, y_pred_custom))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))


# ## Model Interpretation

# In[12]:


coeff = pd.DataFrame({
    "Feature": X.columns,
    "Impact": model.coef_[0]
}).sort_values(by="Impact", ascending=False)

coeff.head(10)


# 
# # Final Conclusion
# 
# ## Summary
# This project successfully implemented a machine learning pipeline to predict customer churn. From data preprocessing to model evaluation, each step was carefully executed to ensure accurate predictions.
# 
# ---
# 
# ## Key Insights
# - Customer service calls and usage patterns strongly influence churn  
# - Certain features have high correlation with customer behavior  
# - Machine learning models can effectively detect churn risk  
# 

# In[ ]:




