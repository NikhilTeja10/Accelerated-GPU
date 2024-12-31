import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df=pd.read_csv(r"MELBOURNE_HOUSE_PRICES_LESS.csv")

df=df.dropna(subset=['Price'])

X=df.drop(columns=['Price'])
y=df['Price']

categorical_features=X.select_dtypes(include=['object']).columns
numerical_features=X.select_dtypes(exclude=['objects']).columns

numerical_transformers=SimpleImputer(strategy='mean')


catgorical_transformers= Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor=ColumnTransformer(
     transformers=[('num',numerical_transformers, numerical_features),
                   ('cat',catgorical_transformers,catgorical_transformers)])


model=Pipeline(steps=[('preprocessor',preprocessor), ('regressor',LinearRegression())])

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_pred,y_test)

print("Mean_squarred_error: ", mse)

print("R-squared:", r2)







