import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import sklearn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

dataset = pd.read_csv("Dataset/LengthOfStay.csv",nrows=1000)
dataset.fillna(0, inplace = True)
dataset.drop(['vdate'], axis = 1,inplace=True)
dataset.drop(['discharged'], axis = 1,inplace=True)

dataset['rcount'] = pd.Series(le1.fit_transform(dataset['rcount'].astype(str)))
dataset['gender'] = pd.Series(le2.fit_transform(dataset['gender'].astype(str)))
dataset['facid'] = pd.Series(le3.fit_transform(dataset['facid'].astype(str)))
print(dataset)

columns = dataset.columns
dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]
print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
'''
scr = ['accuracy']

error = cross_validate(dt, X, Y, cv=10, scoring=scr, return_train_score=True)
print(error)
error = error.get("test_accuracy")
error = np.mean(error)
print(error)
'''

cls = DecisionTreeRegressor(random_state=0)
cls.fit(X_train, y_train)
dt_scores = cross_val_score(cls, X_train, y_train, cv = 10)
print(np.mean(dt_scores))
print(mean_squared_error(cls.predict(X_test), y_test))






