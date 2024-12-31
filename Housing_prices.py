import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV #Randomforesthyperparameters
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv(r'C:\Users\nputta\Downloads\ML_Mid_Term\MELBOURNE_HOUSE_PRICES_LESS.csv')


df = df.dropna(subset=['Price'])


X = df.drop(['Price', 'Suburb', 'Address', 'Date', 'SellerG', 'Regionname', 'CouncilArea'], axis=1, errors='ignore')
y = df['Price']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Add polynomial features
            ('scaler', StandardScaler())
        ]), X.select_dtypes(exclude=['object']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object']).columns)
    ],
    remainder='passthrough'
)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=0))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


param_grid = {
    'regressor__n_estimators': [50, 100],         
    'regressor__max_depth': [10, 15],            
    'regressor__min_samples_split': [5, 10]       
}


print("Scanning the file")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("GridSearchCV completed.")


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Model Hyperparameters:", grid_search.best_params_)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
