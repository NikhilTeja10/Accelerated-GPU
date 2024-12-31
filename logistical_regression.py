import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\nputta\Downloads\ML_Mid_Term\salaries.csv')


print(df.head())
print(df.info())  


X = df.drop('salary_more_then_100k', axis=1)
y = df['salary_more_then_100k']

#creating two sub columns 
df_majority = df[df['salary_more_then_100k'] == 0]
df_minority = df[df['salary_more_then_100k'] == 1]

# Balancing the minority class (salary_more_then_100k = 1) to balance the dataset
df_minority_upsampled = resample(
    df_minority,         
    replace=True,         
    n_samples=len(df_majority),  # Match the number of samples in the majority class
    random_state=0       
)

# Combine the original majority class and the upsampled minority class into one
df_upsampled = pd.concat([
    df_majority,          # Original majority class (salary_more_then_100k = 0)
    df_minority_upsampled # Upsampled minority class (salary_more_then_100k = 1)
])

X = df_upsampled.drop('salary_more_then_100k', axis=1)
y = df_upsampled['salary_more_then_100k']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), X.select_dtypes(include=['object']).columns),  
        ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns)  
    ],
    remainder='passthrough'
)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=0, class_weight='balanced'))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Test Accuracy:", accuracy)


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))

#sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=['<=100k', '>100k'], yticklabels=['<=100k', '>100k'])
sns.heatmap(conf_matrix, annot=True, cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap for Logistic Regression on Salaries Dataset")
plt.show()
