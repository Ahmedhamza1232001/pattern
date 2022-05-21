import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import GridSearchCV
bankdata = pd.read_csv("bill_authentication.csv")
bankdata.shape
bankdata.head()
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
from sklearn.svm import SVC
"""svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))"""
grid = {
    'C':[0.01,0.1,1,10],
    'kernel' : ["linear","poly","rbf","sigmoid"],
    'degree' : [1,3,5,7],
    'gamma' : [0.01,1]
}
svm  = SVC ()
gridcv = GridSearchCV(SVC(), grid)




gridcv.fit(X_train,y_train)

# print best parameter after tuning
print(gridcv.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(gridcv.best_estimator_)
