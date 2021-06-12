#!/usr/bin/env python
# coding: utf-8

# In[303]:


import numpy as np 
import pandas as pd
import mysql.connector

import re

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, LabelBinarizer, Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


get_ipython().run_line_magic('load_ext', 'sql')


# Overview
# The data has been split into two groups:
# 
# * training set (train.csv)
# * test set (test.csv)
# * The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
# 
# * The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
# 
# * We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like

# | Variable | Definition | Key |
# | ------ | ------ | ------ |
# | survival | Survival | 0 = No, 1 = Yes |
# |pclass|Ticket class|1 = 1st, 2 = 2nd, 3 = 3rd|
# |sex|Sex||
# |Age|Age in years||
# |sibsp|# of siblings / spouses aboard the Titanic||
# |parch|# of parents / children aboard the Titanic||
# |ticket|Ticket numbe|
# |fare|Passenger fare|
# |cabin|Cabin numbe|
# |embarke|Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton|

# * Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# * age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# * sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# * parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[304]:


get_ipython().run_line_magic('sql', 'mysql+mysqldb://jwallis:zDxZz4dejpiNhx8F@Mafdet/titanic')


# In[305]:


get_ipython().run_cell_magic('sql', '', 'SELECT * from test LIMIT 10;')


# In[306]:


get_ipython().run_cell_magic('sql', '', 'ALTER TABLE test MODIFY Embarked CHAR(1) NULL;\nALTER TABLE train MODIFY Embarked CHAR(1) NULL;')


# In[307]:


get_ipython().run_cell_magic('sql', '', '#test_cursor = cnx.cursor(buffered = True)\nUPDATE test \nSET NAME = REPLACE(NAME, \'"\', \'\');\n\nUPDATE test \nSET EMBARKED = NULL\nWHERE EMBARKED = \'\\r\';\n\nUPDATE train\nSET NAME = REPLACE(NAME, \'"\', \'\');\n\nUPDATE train \nSET EMBARKED = NULL\nWHERE EMBARKED = \'\\r\';')


# In[308]:


cnx = mysql.connector.connect(user = 'jwallis', password = "zDxZz4dejpiNhx8F", host = "Mafdet", database = 'titanic')
#test = pd.read_sql_query("SELECT * from test", cnx)
test = pd.read_sql("SELECT * from test", cnx)
train = pd.read_sql_query("SELECT * from train", cnx)
gender_submission = pd.read_sql_query("SELECT * from gender_submission", cnx)
cnx.close()

#test["Name"] = [x.strip('"') for x in test["Name"].tolist()]
#train["Name"] = [x.strip('"') for x in train["Name"].tolist()]
print (test.dtypes)
train.head()


# In[309]:


test[["Surname", "Title", "Forename"]] = test["Name"].str.split(r'[.,]', 2, expand = True)
train[["Surname", "Title", "Forename"]] = train["Name"].str.split(r'[.,]', 2, expand = True)
#ex = train["Name"].str.split(r'[.,]', expand = True)
#print (test["Surname"].value_counts())
#test[["title", "split"]] = test["split"].str.split('.', expand = True)
#ex = test["Name"].str.extractall()
#print (ex[3].value_counts())
#ex.where(ex[3] == " Barrett)").dropna()
train.head()


# In[310]:


test.head()


# In[311]:


#train.set_index("PassengerId", inplace = True)
train["Cabin"].replace(to_replace = "", value = np.nan, inplace = True)
test["Cabin"].replace(to_replace = "", value = np.nan, inplace = True)
print (train["Cabin"].unique())
train.head()


# In[ ]:





# In[312]:


gender_submission.head()


# Thoughts:
# * Is logistic regression appropriate here - I am looking for a binary outcome, many of my input features will be discrete, or "labels" rather than continuous variables
# * If I use logistic regression, do I need to render, e.g. sex, into a value (i.e. 1/0)
# 
# Feature engineering
# * Separating out singles/parents/children/children with nannies
# * Separating out cabin to indicate deck --> possible correlation with socioeconomic class/fare
# * worth reviewing titles (particularly for females) for correlation with Parch etc? Identifying children/singles/nannies
# 

# In[313]:


def alpha(entry):
    if entry is not np.nan:
        y = [x for x in entry if x.isalpha() == True][0]
    else:
        y = np.nan
    return y

alpha("A67")

#for a in list(train["Cabin"].unique()):
#    print (a)
#    print (alpha(a))


# In[314]:


train["Cabin Level"] = train["Cabin"].apply(alpha)
test["Cabin Level"] = test["Cabin"].apply(alpha)
X_train = train[["Pclass", "Age", "SibSp"]].copy()
X_test = test[["Pclass", "Age", "SibSp"]].copy()
y_train = train["Survived"].copy()
y_test = gender_submission.copy()
test.head()


# In[316]:


print (train["Cabin Level"].unique())
print (train["Embarked"].unique())
#train.where(train["Embarked"] == "\r").dropna() #practise writing to SQL database from here


# In[296]:


print (np.shape(np.array(train.loc[:, "Embarked"])))
a = np.array(train.loc[:, "Embarked"]).reshape(-1, 1)#[0:10]
print (np.shape(train))
#(x.loc[:, "Embarked"])


# ## Dealing with null values for label encoding

# In[ ]:





# In[318]:


print (train["Embarked"].isna().sum())
print (train["Embarked"].notna().sum())

print (test["Embarked"].isna().sum())
print (test["Embarked"].notna().sum())

#train["Embarked"].value_counts().index[0]
print (train["Cabin Level"].isna().sum())
print (train["Cabin Level"].notna().sum())
print (len(train["Cabin Level"]))
train["Cabin Level"].value_counts()
train["Embarked"].value_counts()


# In[319]:


# fill nans with string values that will be encoded

#train["Sex"].fillna("NAN", inplace = True)
#print 
#train["Embarked"].fillna("NAN", inplace = True)

train["Cabin Level"].fillna("NAN", inplace = True)
test["Cabin Level"].fillna("NAN", inplace = True)
# alternative approaches - fill with most frequently occurring value
# this doesn't work for the cabin level (majority of fields are null)
# for numerical values there is the sklearn imputer function - allows median
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].value_counts().index[0])

#... how do the ML training models handle nan values? how doe 


# In[326]:


sex_encoder = LabelBinarizer() #LabelEncoder()
X_train["SexCat"] = sex_encoder.fit_transform(train.loc[:,"Sex"])
X_test["SexCat"] = sex_encoder.transform(test.loc[:,"Sex"])
#print (sex_encoder.classes_)
sex_codes = pd.DataFrame(index = train["Sex"].unique(), data = sex_encoder.transform(train["Sex"].unique()), columns = ["Code"])
#df["code"] = 
#print (df)

embark_encoder = LabelEncoder()
X_train["Embarked"] = embark_encoder.fit_transform(train.loc[:,"Embarked"])
embark_encoder1hot = OneHotEncoder() #
embarked_1hot = embark_encoder1hot.fit_transform(np.array(X_train.loc[:, "Embarked"]).reshape(-1, 1)) #output is scipy sparse matrix
embarked_codes = pd.DataFrame(index = train["Embarked"].unique(), data = embark_encoder.transform(train["Embarked"].unique()), columns = ["Code 1"])
embarked_codes = pd.DataFrame(index = train["Embarked"].unique(), data = embarked_1hot.transform(np.array(X_train.loc[:, "Embarked"]).reshape(-1, 1))

print (embark_encoder.classes_)

#cabin_encoder = LabelEncoder()
#x["Cabin Level"] = cabin_encoder.fit_transform(train.loc[:,"Cabin Level"])
#print (cabin_encoder.classes_)



#x.head()
sex_codes.head()
print (np.shape(embarked_1hot))
embarked_1hot.toarray()
embarked_codes.head()


# In[15]:


for column in x.columns:
    print (str(column))
    print (x[column].unique())


# In[ ]:





# In[102]:


get_ipython().run_line_magic('sql', 'mysql+mysqldb://jwallis:zDxZz4dejpiNhx8F@Mafdet/titanic')


# In[103]:


get_ipython().run_cell_magic('sql', '', 'SELECT * from test LIMIT 10;')


# In[118]:



SELECT * from test LIMIT 10
#test_cursor.exec


# In[ ]:





# In[22]:


get_ipython().run_cell_magic('sql', '', 'show tables')


# In[23]:


get_ipython().run_cell_magic('sql', '', 'desc gender_submission')


# In[ ]:




