'''
Лабораторная работа №2
Классификация, оценка точности классификации
Для выбранного набора данных необходимо провести сравнение результатов
работы классификаторов из двух групп на выбор:
- регрессия (линейная, нелинейная, логистическая, метод опорных векторов),
k-ближайших соседей;
- деревья принятия решений (деревья, случайный лес);
- бустинг (один из методов на выбор).Для каждого метода необходимо построить матрицу ошибок и рассчитать одну из
следующих метрик: f-мера, ROC AUC, accuracy.
При обучении классификатора необходимо использовать кроссвалидацию.
Необходимо визуализировать изменение точности работы метода на разных шагах
кроссвалидации. Необходимо показать как меняется точность классификации при
изменении гиперпараметров.

Lab #2
Classification, assessment of classification accuracy
For the selected data set, it is necessary to compare the results
work of classifiers from two groups to choose from:
- regression (linear, non-linear, logistic, support vector machine),
k-nearest neighbors;
- decision trees (trees, random forest);
- boosting (one of the methods to choose from). For each method, it is necessary to build an error matrix and calculate one of
following metrics: f-score, ROC AUC, accuracy.
When training a classifier, it is necessary to use cross-validation.
It is necessary to visualize the change in the accuracy of the method at different steps
cross-validation. It is necessary to show how the classification accuracy changes with
changing hyperparameters.
'''

#wine data set:

import pandas as pd

# wine dataset
df = pd.read_csv('wine.csv')

print("\nPreviewing the first few rows of the dataset\n")
print(df.head())
print("\n")

# checking missing values
print(df.isnull().sum())

# To remove any missing value
# df.dropna(inplace=True)

from sklearn.preprocessing import StandardScaler

# scale the data to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('Wine', axis=1))  # scaling all columns except 'Wine'
print("\nscale the data to have a mean of 0 and standard deviation of 1\n")
print(df_scaled)
print("\n")

from sklearn.model_selection import train_test_split

# split the preprocessed dataset into training and testing sets using an 80:20 split ratio
# Wine column as the target variable
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['Wine'], test_size=0.2, random_state=42)

print("\nsplit the preprocessed dataset into training and testing sets\n")
# the shape of the training set is (142, 13), which means that it contains 142 samples (i.e., rows) and 13 features (i.e., columns)
# the shape of the testing set is (36, 13), which means that it contains 36 samples and 13 features
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)
print("\n")



#iris data set:
'''
import pandas as pd

# iris dataset
df = pd.read_csv('iris.csv')

print("\nPreviewing the first few rows of the dataset\n")
print(df.head())
print("\n")



# checking missing values
print(df.isnull().sum())



# To remove any missing value
#df.dropna(inplace=True)



# Converting categorical variables to numerical values
df = pd.get_dummies(df, columns=['variety'], drop_first=True)
print("\nThe dataset after Converting variety column to numerical values\n")
print(df)
print("\n")



from sklearn.preprocessing import StandardScaler
# scale the data to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('Id', axis=1))  # scaling all columns except 'Id'
print("\nscale the data to have a mean of 0 and standard deviation of 1\n")
print("\nfor columns (sepalLength, sepalWidth, petalLength, and petalWidth)\n")
print(df_scaled)
print("\n")


from sklearn.model_selection import train_test_split
#split the preprocessed dataset into training and testing sets using an 80:20 split ratio
#variety_Versicolor column as the target variable
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['variety_Versicolor'], test_size=0.2, random_state=42)

print("\nsplit the preprocessed dataset into training and testing sets\n")
#The shape of the training set is (120, 6), which means that it contains 120 samples (i.e., rows) and 6 features (i.e., columns)

#the shape of the testing set is (30, 6), which means that it contains 30 samples and 6 features
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)
print("\n")
'''
#The target variables y_train and y_test, respectively is (120,), which means that it contains 120 target variables for the corresponding 120 samples in the training set. Similarly, the shape of y_test is (30,), which means that it contains 30 target variables for the corresponding 30 samples in the testing set




# First Group of classifier: will consist of regression and k-nearest neighbors classifiers

# Second Group of classifier: will consist of decision tree, random forest, and boosting classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Instantiate the classifiers
regression = LogisticRegression()
knn = KNeighborsClassifier()

decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
boosting = GradientBoostingClassifier()


from sklearn.model_selection import cross_val_score

#Train and evaluate classifiers
regression_scores = cross_val_score(regression, X_train, y_train, cv=5)
print("Cross-validation regression_scores:\n", regression_scores)
# print("Average score:", regression_scores.mean())
# print("Standard deviation:", regression_scores.std())

knn_scores = cross_val_score(knn, X_train, y_train, cv=5)
print("\nCross-validation knn_scores:\n", knn_scores)

decision_tree_scores = cross_val_score(decision_tree, X_train, y_train, cv=5)
print("\nCross-validation decision_tree_scores:\n", decision_tree_scores)

random_forest_scores = cross_val_score(random_forest, X_train, y_train, cv=5)
print("\nCross-validation random_forest_scores:\n", random_forest_scores)

boosting_scores = cross_val_score(boosting, X_train, y_train, cv=5)
print("\nCross-validation boosting_scores:\n", boosting_scores)
#cross-validation (CV) is a technique for evaluating how well a trained model can generalize to new, unseen data
#cv=5 means that the dataset will be divided into 5 folds

#These scores can be used to compare the performance of the classifiers and choose the best one for our task




from sklearn.metrics import confusion_matrix, accuracy_score
#regression
print("\n")
regression = LogisticRegression() # Predict the labels of the testing set using the logistic regression classifier
regr = regression.fit(X_train, y_train)
lr_predictions = regr.predict(X_test)

lr_cm = confusion_matrix(y_test, lr_predictions) # Create a confusion matrix to evaluate the performance of the logistic regression classifier
print("Logistic regression confusion matrix:")
print(lr_cm)

lr_accuracy = accuracy_score(y_test, lr_predictions) # Calculate the accuracy of the logistic regression classifier
print("Logistic regression accuracy:", lr_accuracy)
print("\n")

#k-nearest neighbors
print("\n")
knn = KNeighborsClassifier()
knnf = knn.fit(X_train, y_train)
kn_predictions = knnf.predict(X_test)

kn_cm = confusion_matrix(y_test, kn_predictions)
print("Logistic k-nearest neighbors confusion matrix:")
print(kn_cm)

kn_accuracy = accuracy_score(y_test, kn_predictions)
print("Logistic k-nearest neighbors accuracy:", kn_accuracy)
print("\n")

#decision_tree
print("\n")
decision_tree = DecisionTreeClassifier()
Treef = decision_tree.fit(X_train, y_train)
Tr_predictions = Treef.predict(X_test)

Tr_cm = confusion_matrix(y_test, Tr_predictions)
print("Logistic decision_tree confusion matrix:")
print(Tr_cm)

Tr_accuracy = accuracy_score(y_test, Tr_predictions)
print("Logistic decision_tree accuracy:", Tr_accuracy)
print("\n")

#random_forest
print("\n")
random_forest = RandomForestClassifier()
Ranf = random_forest.fit(X_train, y_train)
Ran_predictions = Ranf.predict(X_test)

Ran_cm = confusion_matrix(y_test, Ran_predictions)
print("Logistic random_forest confusion matrix:")
print(Ran_cm)

Ran_accuracy = accuracy_score(y_test, Ran_predictions)
print("Logistic random_forest accuracy:", Ran_accuracy)
print("\n")

#boosting
print("\n")
boosting = GradientBoostingClassifier()
Boof = boosting.fit(X_train, y_train)
Boo_predictions = Boof.predict(X_test)

Boo_cm = confusion_matrix(y_test, Boo_predictions)
print("Logistic boosting confusion matrix:")
print(Boo_cm)

Boo_accuracy = accuracy_score(y_test, Boo_predictions)
print("Logistic boosting accuracy:", Boo_accuracy)
print("\n")


#//////////////////////////////////////////////////////
#//////////////////////////////////////////////////////
#//////////////////////////////////////////////////////
#Visualize the accuracy
#//////////////////////////////////////////////////////


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

#regression
scores = cross_val_score(regression, X_train, y_train, cv=10)
print("Cross-validation scores - regression:\n", scores)
plt.plot(np.arange(1, 11), scores, '-o')
plt.xlabel('Cross-validation fold')
plt.ylabel('Accuracy')
plt.title('Cross-validation scores - regression')
plt.show()
print("\n")
#k-nearest neighbors
scores = cross_val_score(knn, X_train, y_train, cv=10)
print("Cross-validation scores - k-nearest neighbors:\n", scores)
plt.plot(np.arange(1, 11), scores, '-o')
plt.xlabel('Cross-validation fold')
plt.ylabel('Accuracy')
plt.title('Cross-validation scores - k-nearest neighbors')
plt.show()
print("\n")
#decision_tree
scores = cross_val_score(decision_tree, X_train, y_train, cv=10)
print("Cross-validation scores - decision_tree:\n", scores)
plt.plot(np.arange(1, 11), scores, '-o')
plt.xlabel('Cross-validation fold')
plt.ylabel('Accuracy')
plt.title('Cross-validation scores - decision_tree')
plt.show()
print("\n")
#random_forest
scores = cross_val_score(random_forest, X_train, y_train, cv=10)
print("Cross-validation scores - random_forest:\n", scores)
plt.plot(np.arange(1, 11), scores, '-o')
plt.xlabel('Cross-validation fold')
plt.ylabel('Accuracy')
plt.title('Cross-validation scores - random_forest')
plt.show()
print("\n")
#boosting
scores = cross_val_score(boosting, X_train, y_train, cv=10)
print("Cross-validation scores - boosting:", scores)
plt.plot(np.arange(1, 11), scores, '-o')
plt.xlabel('Cross-validation fold')
plt.ylabel('Accuracy')
plt.title('Cross-validation scores - boosting')
plt.show()
print("\n")

#/////////////////////////////////////////////////////////////
#grid search to tune the hyperparameters of the random forest classifier:
print("\n")
from sklearn.model_selection import GridSearchCV

#Regression
#parameter grid
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
# param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
# param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50]}
# param_grid = {'C': np.logspace(-3, 3, 7)}
lr = LogisticRegression()
#grid search with 5-fold cross-validation
grid_search = GridSearchCV(lr, param_grid, cv=5) #cv=5
grid_search.fit(X_train, y_train)
print("\nRegression")
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.best_score_)
print("\n")
#///////////////Test/////////////////////
import matplotlib.pyplot as plt
# extract results of grid search
# c_values = [0.1, 0.5, 1, 5, 10, 50]
c_values = [0.00001, 0.0001, 0.1, 1, 10, 100]
scores = grid_search.cv_results_['mean_test_score']
# plot results
plt.plot(c_values, scores, '-o')
plt.xscale('log')
plt.xlabel('C value')
plt.ylabel('Cross-validation score')
plt.title('Grid search results')
plt.show()
#////////////////////////////////////////

#k-nearest neighbors [1, 3, 5, 7, 9]
param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}
lr = KNeighborsClassifier()
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("\nk-nearest neighbors")
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.best_score_)
print("\n")

#decision_tree max_depth [2, 5, 10, None]
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [5, 8, 10],
    'min_samples_leaf': [3, 4, 6]
}
#   'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]

lr = DecisionTreeClassifier()
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("\ndecision_tree")
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.best_score_)
print("\n")

#RandomForest
param_grid = {'n_estimators': [50, 100],
              'max_depth': [5, 10],
              'min_samples_split': [2, 5],
              'min_samples_leaf': [1, 2],
              'max_features': ['sqrt', 'log2', None]}
lr = RandomForestClassifier()
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("\nRandomForest")
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.best_score_)
print("\n")

#GradientBoosting
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'max_features': ['sqrt']
}
lr = GradientBoostingClassifier()
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("\nGradientBoosting")
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.best_score_)
print("\n")
