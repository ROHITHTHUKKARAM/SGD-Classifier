# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1:Import Necessary Libraries and Load Data

step 2:Split Dataset into Training and Testing Sets

step 3:Train the Model Using Stochastic Gradient Descent (SGD)

step 4:Make Predictions and Evaluate Accuracy 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Rohith T
RegisterNumber: 212223040173

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data= iris.data,columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
*/
```

## Output:
![Screenshot 2024-09-19 091901](https://github.com/user-attachments/assets/8a89873e-dd91-4bbf-b918-7eabbec9bace)


![Screenshot 2024-09-19 091909](https://github.com/user-attachments/assets/760631b1-68bf-49fd-98d7-7614247ca2d6)


![Screenshot 2024-09-19 091921](https://github.com/user-attachments/assets/c4e13ce8-a4f0-4af9-9772-e2ed43ef5ac0)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
