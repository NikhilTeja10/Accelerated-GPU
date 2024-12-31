from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


digits = datasets.load_digits()


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=0)

#hyperparameters 
kernels = ['linear', 'rbf', 'poly']  #decision boundary 
C_values = [0.1, 1, 10]     #accuracy           
gamma_values = ['scale', 0.001, 0.01, 0.1, 1]  # influ training points 

best_accuracy = 0
best_params = {}
results = []

for kernel in kernels:
    for C in C_values:
        for gamma in gamma_values:
            svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=0)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((kernel, C, gamma, accuracy))
            print(f"Kernel: {kernel}, C: {C}, Gamma: {gamma} --> Accuracy: {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}

print("\nBest Configuration:")
print(f"Kernel: {best_params['kernel']}, C: {best_params['C']}, Gamma: {best_params['gamma']} --> Accuracy: {best_accuracy:.4f}")


best_svm = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], random_state=0)
best_svm.fit(X_train, y_train)
y_pred_best = best_svm.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap for Best SVM Model on Digits Dataset")
plt.show()
