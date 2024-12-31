import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'titanic.csv')

#irrelevant 
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df = df.dropna()

#converting categorical to numerical 
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


X = df[['Pclass', 'Age', 'Fare', 'Sex']]  
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#gaussian normal distribution 
model = GaussianNB()
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("Naive Bayes Test Accuracy:", accuracy)


y_pred = model.predict(X_test[:10])
print("First 10 Predictions:", y_pred)


print("First 10 Actual Labels:", y_test[:10].values)


y_proba = model.predict_proba(X_test[:10])
print("Prediction Probabilities for First 10 Samples:\n", y_proba)


conf_matrix = confusion_matrix(y_test, model.predict(X_test))
print("Confusion Matrix:\n", conf_matrix)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap for Naive Bayes on Titanic Dataset")
plt.show()

