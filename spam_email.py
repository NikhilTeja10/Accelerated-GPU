import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\nputta\Downloads\ML_Mid_Term\spam.csv', encoding='latin-1')


print(df.head())
print(df.columns)

#necessary columns 
df = df[['Category', 'Message']]

#encoding binary  
df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})


X = df['Message']
y = df['Category']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


vectorizer = CountVectorizer() #strings to numeric values 
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


nb_model = MultinomialNB() #naive bayes work well with word counts 
nb_model.fit(X_train_vec, y_train)


y_pred = nb_model.predict(X_test_vec)


accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Spam Classifier Accuracy:", accuracy)


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for Naive Bayes Spam Classifier")
plt.show()


