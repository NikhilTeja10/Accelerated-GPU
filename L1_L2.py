import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


df = pd.read_csv(r'C:\Users\nputta\Downloads\ML_Mid_Term\MELBOURNE_HOUSE_PRICES_LESS.csv')


df = df.dropna(subset=['Price'])


X = df.drop(['Price', 'Suburb', 'Address', 'Date', 'SellerG', 'Regionname', 'CouncilArea'], axis=1, errors='ignore')
y = df['Price']


X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


preprocessor = ColumnTransformer([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False), X.columns),
    ('scaler', StandardScaler(), X.columns)
], remainder='passthrough')


models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression (L1)": Lasso(alpha=10, max_iter=10000, tol=0.01), 
    "Ridge Regression (L2)": Ridge(alpha=10, max_iter=10000, tol=0.01)   
}


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    print(f"\n{name} R-squared:")
    print("Training R^2:", train_score)
    print("Test R^2:", test_score)

