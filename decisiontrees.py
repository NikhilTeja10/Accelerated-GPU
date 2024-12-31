import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

#Overall dataset
df = pd.read_csv(r'C:\Users\nputta\Downloads\ML_Mid_Term\titanic.csv')


print(df.head())


df = df.dropna(subset=['Survived'])

#Features
X = df.drop('Survived', axis=1)
#Target data
y = df['Survived']

#Numerical 
for column in X.select_dtypes(include=['float64', 'int64']).columns:
    X[column].fillna(X[column].median(), inplace=True)

#Categorical 
for column in X.select_dtypes(include=['object']).columns:
    X[column].fillna(X[column].mode()[0], inplace=True)

# preprocessing step 
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), X.select_dtypes(include=['object']).columns),  # Avoid multicollinearity
        ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns)  # Scale numerical columns
    ],
    remainder='passthrough'
)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=0))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Test Accuracy:", accuracy)


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap for Decision Tree on Titanic Dataset")
plt.show()
