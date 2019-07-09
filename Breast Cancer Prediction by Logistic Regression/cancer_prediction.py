# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing our cancer dataset
dataset = pd.read_csv('breast_cancer_dataset.csv')
X = dataset.iloc[:, 1:9].values
Y = dataset.iloc[:, 9].values

# Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Fitting Simple Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression( solver='lbfgs', max_iter=500)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm_SVM = confusion_matrix(Y_test, Y_pred)
print(cm_SVM)
print("Accuracy score of train Classifier")
print(accuracy_score(Y_train, classifier.predict(X_train))*100)
print("Accuracy score of test Classifier")
print(accuracy_score(Y_test, Y_pred)*100)

