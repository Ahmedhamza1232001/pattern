
#importing necessary libraries
import numpy as np
from sklearn.naive_bayes import BernoulliNB 
from sklearn.metrics  import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
#creating dataset with with 0 and 1
X = np.random.randint(2, size=(500,10))
Y = np.random.randint(2, size=(500, 1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
clf = BernoulliNB()
model = clf.fit(X, Y)
y_pred =clf.predict(X_test)
acc_score= accuracy_score(y_test, y_pred)
print(acc_score)