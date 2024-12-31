from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


iris = load_iris()
X, y = iris.data, iris.target


kf = KFold(n_splits=5, shuffle=True, random_state=0)  


def cross_val_score_model(model, X, y):
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    
    return np.mean(scores)


lr = LogisticRegression(max_iter=1000)
lr_score = cross_val_score_model(lr, X, y)
print("Logistic Regression Average Test Score (5-Fold CV):", lr_score)


svm = SVC()
svm_score = cross_val_score_model(svm, X, y)
print("SVM Average Test Score (5-Fold CV):", svm_score)

rf = RandomForestClassifier(n_estimators=40)
rf_score = cross_val_score_model(rf, X, y)
print("Random Forest Average Test Score (5-Fold CV):", rf_score)
