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


"""
plt.figure(figsize=(9,4))
plt.bar(range(17),explained_variance)
plt.title('PCA-Column Contribution')
plt.xlabel('Column Name')
plt.ylabel('variance ration')
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Logistic Regression to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Applying the k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
#10 accuracy will be returned that will be computed through 10 computation using k-fold
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)

#Take the average or mean on accuracies
mean_accuracies=accuracies.mean()
std_accuracies=accuracies.std()*100







"""
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red',  'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

"""




