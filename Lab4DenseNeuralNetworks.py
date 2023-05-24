import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# The Perceptron is one of the earliest and simplest forms of artificial neural networks. It is a binary classifier used for supervised learning tasks. The Perceptron algorithm was developed by Frank Rosenblatt in the 1950s and is based on the concept of a single-layer neural network
from sklearn.decomposition import PCA

df = pd.read_csv('wine.csv')

print("|||||||||||||||||||||||||||||||||||||||||||||||||")
print("\nPreviewing the first few rows of the dataset\n")
print("|||||||||||||||||||||||||||||||||||||||||||||||||")
print(df.head())
print("\n")
print(df.isnull().sum())

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('Wine', axis=1))

print("|||||||||||||||||||||||||||||||||||||||||||||||||")
print("\nScale the data\n")
print("|||||||||||||||||||||||||||||||||||||||||||||||||")
print(df_scaled)
print("\n")

X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['Wine'], test_size=0.2, random_state=42)

print("\nSplit the preprocessed dataset into training and testing sets\n")
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)
print("\n")

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
perceptron_predictions = perceptron.predict(X_test)
perceptron_accuracy = accuracy_score(y_test, perceptron_predictions)
print("Perceptron accuracy:", perceptron_accuracy)

regression = LogisticRegression()
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
boosting = GradientBoostingClassifier()

regression_scores = cross_val_score(regression, X_train, y_train, cv=5)
knn_scores = cross_val_score(knn, X_train, y_train, cv=5)
decision_tree_scores = cross_val_score(decision_tree, X_train, y_train, cv=5)
random_forest_scores = cross_val_score(random_forest, X_train, y_train, cv=5)
boosting_scores = cross_val_score(boosting, X_train, y_train, cv=5)

print("\nCross-validation regression scores:", regression_scores)
print("\nCross-validation knn scores:", knn_scores)
print("\nCross-validation decision tree scores:", decision_tree_scores)
print("\nCross-validation random forest scores:", random_forest_scores)
print("\nCross-validation boosting scores:", boosting_scores)
print('\n')

plt.plot(np.arange(1, 6), regression_scores, '-o', label='Logistic Regression')
plt.plot(np.arange(1, 6), knn_scores, '-o', label='K-nearest Neighbors')
plt.plot(np.arange(1, 6), decision_tree_scores, '-o', label='Decision Tree')
plt.plot(np.arange(1, 6), random_forest_scores, '-o', label='Random Forest')
plt.plot(np.arange(1, 6), boosting_scores, '-o', label='Boosting')
plt.plot(np.arange(1, 6), [perceptron_accuracy] * 5, '-o', label='Perceptron') 
plt.xlabel('Cross-validation fold')
plt.ylabel('Accuracy')
plt.title('Cross-validation scores')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['Wine'], test_size=0.2, random_state=42)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
perceptron_predictions = perceptron.predict(X_test)
perceptron_accuracy = accuracy_score(y_test, perceptron_predictions)
print("Perceptron accuracy:", perceptron_accuracy)

boosting = GradientBoostingClassifier()
boosting.fit(X_train, y_train)
boosting_predictions = boosting.predict(X_test)
boosting_accuracy = accuracy_score(y_test, boosting_predictions)
print("Gradient Boosting accuracy:", boosting_accuracy)

pca_perceptron = PCA(n_components=2)
X_train_pca_perceptron = pca_perceptron.fit_transform(X_train)

pca_boosting = PCA(n_components=2)
X_train_pca_boosting = pca_boosting.fit_transform(X_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(X_train_pca_perceptron[y_train == 1, 0], X_train_pca_perceptron[y_train == 1, 1], label='Cluster 1')
axes[0].scatter(X_train_pca_perceptron[y_train == 2, 0], X_train_pca_perceptron[y_train == 2, 1], label='Cluster 2')
axes[0].scatter(X_train_pca_perceptron[y_train == 3, 0], X_train_pca_perceptron[y_train == 3, 1], label='Cluster 3')
axes[0].set_xlabel('Component 1')
axes[0].set_ylabel('Component 2')
axes[0].set_title('PCA Clusters for Perceptron')
axes[0].legend()

axes[1].scatter(X_train_pca_boosting[y_train == 1, 0], X_train_pca_boosting[y_train == 1, 1], label='Cluster 1')
axes[1].scatter(X_train_pca_boosting[y_train == 2, 0], X_train_pca_boosting[y_train == 2, 1], label='Cluster 2')
axes[1].scatter(X_train_pca_boosting[y_train == 3, 0], X_train_pca_boosting[y_train == 3, 1], label='Cluster 3')
axes[1].set_xlabel('Component 1')
axes[1].set_ylabel('Component 2')
axes[1].set_title('PCA Clusters for Gradient Boosting')
axes[1].legend()

plt.tight_layout()
plt.show()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////

X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['Wine'], test_size=0.2, random_state=42)
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'K-nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Perceptron': Perceptron()
}
accuracy_scores = {}
for clf_name, clf in classifiers.items():
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    accuracy_scores[clf_name] = scores.mean()

plt.figure(figsize=(10, 6))
x_pos = np.arange(len(accuracy_scores))
plt.bar(x_pos, list(accuracy_scores.values()), align='center', alpha=0.8)
plt.xticks(x_pos, list(accuracy_scores.keys()), rotation=45)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Comparison of Classifier Accuracies')
plt.tight_layout()
plt.show()


# how many hidden layers has your perceptron?

# single-layer neural network, the input features are directly connected to the output layer, and there are no intermediate hidden layers