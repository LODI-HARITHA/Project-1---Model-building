######################## KNN ##########################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}  # Example values for n_neighbors

# Initialize and train the KNN classifier
k = 5  # Number of neighbors
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


############################### SVM ################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
model = SVC()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)



################################# Random Forests #################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

############################## Linear Regression ###################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


############################## Logistic Regression ###########################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


################################ Naive Bayes #########################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes model (GaussianNB)
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

######################## PCA #########################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)  # Specify the number of components
X_pca = pca.fit_transform(X_scaled)

# Split the PCA-transformed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize and train the KNN classifier
k = 5  # Number of neighbors
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)



import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Preprocess the data if needed
# ...

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Perform hierarchical clustering
n_clusters = 3  # Specify the number of clusters
model = AgglomerativeClustering(n_clusters=n_clusters)
clusters = model.fit_predict(X)

# Visualize the clustering results (optional)
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hierarchical Clustering')
plt.show()


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'accuracy',
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Create the DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train the XGBoost model
model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions on the training and testing data
y_train_pred = model.predict(dtrain)
y_test_pred = model.predict(dtest)

# Convert predicted probabilities to binary predictions
y_train_pred_binary = [1 if p >= 0.5 else 0 for p in y_train_pred]
y_test_pred_binary = [1 if p >= 0.5 else 0 for p in y_test_pred]

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred_binary)
test_accuracy = accuracy_score(y_test, y_test_pred_binary)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)



import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}  # Example values for n_neighbors

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the testing data using the best model
y_pred = best_model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Best Parameters:", best_params)
print("Testing Accuracy:", accuracy)


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}  # Example values for C and gamma

# Initialize the SVM classifier
svm = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the testing data using the best model
y_pred = best_model.predict(X_test)
y_pred = best_model.predict(X_train)
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Best Parameters:", best_params)
print("Testing Accuracy:", accuracy)
print('Training accuracy:',accuracy)



import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
hyperparameters = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# Initialize the Random Forest classifier
model = RandomForestClassifier()

# Perform grid search cross-validation to find the best parameters
grid_search = GridSearchCV(model, hyperparameters, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model with the best parameters
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
hyperparameters = {
    "normalize": [True, False]
}

# Initialize the Linear Regression model
model = LinearRegression()

# Perform grid search cross-validation to find the best parameters
grid_search = GridSearchCV(model, hyperparameters, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model with the best parameters
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate mean squared error
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)




import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
hyperparameters = {
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

# Initialize the Logistic Regression model
model = LogisticRegression()

# Perform grid search cross-validation to find the best parameters
grid_search = GridSearchCV(model, hyperparameters, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model with the best parameters
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)









import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read the data into a pandas DataFrame
data = pd.read_excel("C:/Users/DELL PC/cleaned data.xlsx")

# Split the data into features (X) and target variable (y)
X = data.drop("Failure_status", axis=1)
y = data["Failure_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
hyperparameters = {
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

# Initialize the Logistic Regression model
model = LogisticRegression()

# Perform grid search cross-validation to find the best parameters
grid_search = GridSearchCV(model, hyperparameters, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model with the best parameters
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)












