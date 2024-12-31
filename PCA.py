import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv(r'heart.csv')  

print(df.shape)

#categorical to numerical 
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])


X = df.drop('HeartDisease', axis=1)  
y = df['HeartDisease'] 


#m=0 v=1 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#2 dimensional dataset
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)



print("Explained Variance Ratio:", pca.explained_variance_ratio_)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}


for name, clf in classifiers.items():

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{name} Classifier Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
   
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"{name} Confusion Matrix:\n{conf_matrix}")


#1. jupyter 
#2. email , corected (split before and after pca)
#3. 3 run the cleaned data and see check reduced data 
#4. every category, version 3 get dummies for all the variaables, then accuracy 