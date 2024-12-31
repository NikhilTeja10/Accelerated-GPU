import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)


y = X['petal length (cm)']
X = X.drop(columns=['petal length (cm)'])


model = LinearRegression()


k = 5  
kf = KFold(n_splits=k, shuffle=True, random_state=0)


mse_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    

    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    

    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print("Mean MSE across all folds:", np.mean(mse_scores))




mse_scores_cross_val = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
print("Mean MSE using cross_val_score:", -np.mean(mse_scores_cross_val))
