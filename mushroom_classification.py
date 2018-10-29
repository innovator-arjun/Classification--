# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:05:23 2018

@author: avaithil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset 
dataset=pd.read_csv('mushrooms.csv')

#Check for null or nan in the dataset
dataset.isnull().sum()

from sklearn.preprocessing import LabelEncoder

#Encode the labeled data since it is a categorical variable
label=LabelEncoder()

for col in dataset.columns:
    dataset[col]=label.fit_transform(dataset[col])
    
dataset.head()


dataset['class'].value_counts().plot.bar()



#Spliting up the data into independent and dependent variable
X=dataset.iloc[:,1:24].values
y=dataset.iloc[:,0].values

corr_table=dataset.corr()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)



from sklearn.decomposition import PCA
pca=PCA(n_components=17,random_state=0)
X=pca.fit_transform(X)
explained_variance=pca.explained_variance_ratio_



plt.figure(figsize=(9,4))
plt.bar(range(17),explained_variance)
plt.title('PCA-Column Contribution')
plt.xlabel('Column Name')
plt.ylabel('variance ration')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy
(820+723)/(820+32+50+723)

