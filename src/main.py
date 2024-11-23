# Author: Pranav Mehrotra

import pandas as pd

import numpy as np

import time

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler


# Reading the data and preprocessing it:

df = pd.read_csv('../data/all_data.csv')

X = df.drop('Label', axis=1)

X = StandardScaler().fit_transform(X)

Y = df["Label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# Training a Random Forest Machine Learning Model

start_time = time.time()

rf_classifier_1 = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier_1.fit(X_train, Y_train)

end_time = time.time()

training_time_1 = round(end_time - start_time,2)

# Testing accuracy of the model

Y_pred = rf_classifier_1.predict(X_test)

accuracy_1 = round(accuracy_score(Y_test, Y_pred)*100, 2)


# Implemening Principal Component Analysis

covariance_matrix = np.cov(X.T)

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

variance_explained = [(i / sum(eigen_values))*100 for i in eigen_values]

cumulative_variance_explained = np.cumsum(variance_explained)

total_selected_columns = 0

for i in cumulative_variance_explained:

    if (i > 95):

        total_selected_columns = cumulative_variance_explained.tolist().index(i) + 1

        break

projection_matrix = (eigen_vectors.T[:][:])[:(total_selected_columns - 1)].T

X_train = np.real(X_train.dot(projection_matrix))

X_test = np.real(X_test.dot(projection_matrix))


# Training a Random Forest Machine Learning Model again

start_time = time.time()

rf_classifier_2 = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier_2.fit(X_train, Y_train)

end_time = time.time()

training_time_2 = round(end_time - start_time,2)

# Testing accuracy of the model

Y_pred = rf_classifier_2.predict(X_test)

accuracy_2 = round(accuracy_score(Y_test, Y_pred)*100,2)

# Storing the statistics for both the models

with open("../output/result.txt", 'w') as file:
    
    file.write(f"Accuracy on ML_model_1 (before PCA Implementation): {accuracy_1}%\n")
    file.write(f"Time to train ML_model_1 (before PCA Implementation): {training_time_1} seconds\n\n")
    file.write(f"Accuracy on ML_model_1 (after PCA Implementation): {accuracy_2}%\n")
    file.write(f"Time to train ML_model_2 (after PCA Implementation): {training_time_2} seconds\n\n")
    file.write("Percentage decrease in running Time after PCA Implementation: " + str(round((training_time_2/training_time_1)*100)) + "%")

