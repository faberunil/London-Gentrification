import sys
import os

# Get the path to the parent directory (main folder)
parent_directory = os.path.abspath('.')

# Add the parent directory to the Python path
sys.path.append(parent_directory)

import pandas as pd
from sklearn.model_selection import train_test_split # For splitting data
from sklearn.linear_model import LogisticRegression # Logistic Regression Classifier
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors Classifier
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boosting Classifier
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score # For evaluation
from sklearn.utils import resample  # For upsampling
from sklearn.model_selection import GridSearchCV # For hyperparameter tuning
import joblib  # For saving models

import EDA.df_variations as dv  # all the necessary DataFrames

# Define bins and labels for the classification
bins = [0, 0.2, 0.4, 0.6, 1]
labels = ['Low/No Gentrification', 'Mild Gentrification', 'High Gentrification', 'Extreme Gentrification']

# Helper functions
def prepare_data(df, index_col, drop_cols):
    """Prepare data by splitting into training and testing sets and optionally applying upsampling."""
    df = df[df['Year'] != 2010]
    df['gentrification_category'] = pd.cut(df[index_col], bins=bins, labels=labels, include_lowest=True)
    X = df.drop(drop_cols + ['gentrification_category'], axis=1)
    y = df['gentrification_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    return X_train, X_test, y_train, y_test

def apply_upsampling(df, index_col, drop_cols):
    """Apply upsampling to balance the dataset."""
    # Filter the DataFrame and avoid in-place modifications
    filtered_df = df[df['Year'] != 2010].copy()
    filtered_df['gentrification_category'] = pd.cut(filtered_df[index_col], bins=bins, labels=labels, include_lowest=True)
    
    # Prepare features and target
    X = filtered_df.drop(drop_cols, axis=1)
    y = filtered_df['gentrification_category']
    
    # Split into train and test to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    
    # Combine X_train and y_train for resampling
    train_data = pd.concat([X_train, y_train], axis=1)
    majority_class = train_data[train_data.gentrification_category == 'High Gentrification']
    minority_classes = [train_data[train_data.gentrification_category == label] for label in labels if label != 'High Gentrification']
    
    # Resample each minority class to the size of the majority class
    minority_upsampled = [resample(minority,
                                   replace=True,  # sample with replacement
                                   n_samples=len(majority_class),  # to match majority class size
                                   random_state=42) for minority in minority_classes]
    
    # Combine majority class with upsampled minority classes
    upsampled_train = pd.concat([majority_class] + minority_upsampled)
    y_train_upsampled = upsampled_train['gentrification_category']
    X_train_upsampled = upsampled_train.drop('gentrification_category', axis=1)
    
    return X_train_upsampled, X_test, y_train_upsampled, y_test

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Model", "DataFrame", "Accuracy", "Precision", "Recall", "F1-Score"])

def update_results(model_name, df_name, y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    results_df.loc[len(results_df)] = [model_name, df_name, accuracy, precision, recall, f1]
###############################################################################
# Logistic Regression (with summarized_df) ####################################
###############################################################################
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = apply_upsampling(
    dv.summarized_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White']
)
logisticsum_model = LogisticRegression(max_iter=1000, C=100, solver='liblinear')
logisticsum_model.fit(X_train_logistic, y_train_logistic)
predictions_logisticsum = logisticsum_model.predict(X_test_logistic)
print("Summarized Logistic Regression Classification Report:")
print(classification_report(y_test_logistic, predictions_logisticsum))
print("Accuracy:", accuracy_score(y_test_logistic, predictions_logisticsum))
update_results("Logistic Regression", "Summarized", y_test_logistic, predictions_logisticsum)

###############################################################################
# Logistic Regression with rate_change_df ####################################
###############################################################################
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = apply_upsampling(
    dv.rate_change_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White Rate Change']
)
logisticrc_model = LogisticRegression(max_iter=1000, C=100, solver='liblinear')
logisticrc_model.fit(X_train_logistic, y_train_logistic)
predictions_logisticrc = logisticrc_model.predict(X_test_logistic)
print("Rate Change Logistic Regression Classification Report:")
print(classification_report(y_test_logistic, predictions_logisticrc))
print("Accuracy:", accuracy_score(y_test_logistic, predictions_logisticrc))
update_results("Logistic Regression", "Rate Change", y_test_logistic, predictions_logisticrc)

###############################################################################
# Logistic Regression with normalized_df #####################################
###############################################################################
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = apply_upsampling(
    dv.normalized_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White Normalized']
)
logisticnorm_model = LogisticRegression(max_iter=1000, C=100, solver='lbfgs')
logisticnorm_model.fit(X_train_logistic, y_train_logistic)
predictions_logisticnorm = logisticnorm_model.predict(X_test_logistic)
print("Normalized Logistic Regression Classification Report:")
print(classification_report(y_test_logistic, predictions_logisticnorm))
print("Accuracy:", accuracy_score(y_test_logistic, predictions_logisticnorm))
update_results("Logistic Regression", "Normalized", y_test_logistic, predictions_logisticnorm)

###############################################################################
# KNN with summarized_df ######################################################
###############################################################################
X_train_knn, X_test_knn, y_train_knn, y_test_knn = apply_upsampling(
    dv.summarized_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White']
)
knnsum_model = KNeighborsClassifier(n_neighbors=10, weights='distance')
knnsum_model.fit(X_train_knn, y_train_knn)
predictions_knnsum = knnsum_model.predict(X_test_knn)
print("Summarized KNN Classification Report:")
print(classification_report(y_test_knn, predictions_knnsum))
print("Accuracy:", accuracy_score(y_test_knn, predictions_knnsum))
update_results("KNN", "Summarized", y_test_knn, predictions_knnsum)

###############################################################################
# KNN with rate_change_df #####################################################
###############################################################################
X_train_knn, X_test_knn, y_train_knn, y_test_knn = apply_upsampling(
    dv.rate_change_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White Rate Change']
)
knnrc_model = KNeighborsClassifier(n_neighbors=10, weights='distance')
knnrc_model.fit(X_train_knn, y_train_knn)
predictions_knnrc = knnrc_model.predict(X_test_knn)
print("Rate Change KNN Classification Report:")
print(classification_report(y_test_knn, predictions_knnrc))
print("Accuracy:", accuracy_score(y_test_knn, predictions_knnrc))
update_results("KNN", "Rate Change", y_test_knn, predictions_knnrc)

###############################################################################
# KNN with normalized_df ######################################################
###############################################################################
X_train_knn, X_test_knn, y_train_knn, y_test_knn = apply_upsampling(
    dv.normalized_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White Normalized']
)
knnnorm_model = KNeighborsClassifier(n_neighbors=10, weights='distance')
knnnorm_model.fit(X_train_knn, y_train_knn)
predictions_knnnorm = knnnorm_model.predict(X_test_knn)
print("Normalized KNN Classification Report:")
print(classification_report(y_test_knn, predictions_knnnorm))
print("Accuracy:", accuracy_score(y_test_knn, predictions_knnnorm))
update_results("KNN", "Normalized", y_test_knn, predictions_knnnorm)

###############################################################################
# Random Forest with summarized_df ############################################
###############################################################################
X_train_rf, X_test_rf, y_train_rf, y_test_rf = apply_upsampling(
    dv.summarized_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White']
)
rfsum_model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=10, min_samples_leaf= 1, min_samples_split=5)
rfsum_model.fit(X_train_rf, y_train_rf)
predictions_rfsum = rfsum_model.predict(X_test_rf)
print("Summarized Random Forest Classification Report (with upsampling):")
print(classification_report(y_test_rf, predictions_rfsum))
print("Accuracy:", accuracy_score(y_test_rf, predictions_rfsum))
update_results("Random Forest", "Summarized", y_test_rf, predictions_rfsum)

###############################################################################
# Random Forest with rate_change_df ###########################################
###############################################################################
X_train_rf, X_test_rf, y_train_rf, y_test_rf = apply_upsampling(
    dv.rate_change_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White Rate Change']
)
rfrc_model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=10, min_samples_leaf= 1, min_samples_split=5)
rfrc_model.fit(X_train_rf, y_train_rf)
predictions_rfrc = rfrc_model.predict(X_test_rf)
print("Rate Change Random Forest Classification Report (with upsampling):")
print(classification_report(y_test_rf, predictions_rfrc))
print("Accuracy:", accuracy_score(y_test_rf, predictions_rfrc))
update_results("Random Forest", "Rate Change", y_test_rf, predictions_rfrc)
# Save RFrc model
joblib.dump(rfrc_model, 'Models/rfrc_model.pkl')

###############################################################################
# Random Forest with normalized_df ###########################################
###############################################################################
X_train_rf, X_test_rf, y_train_rf, y_test_rf = apply_upsampling(
    dv.normalized_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White Normalized']
)
rfnorm_model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=10, min_samples_leaf= 1, min_samples_split=5)
rfnorm_model.fit(X_train_rf, y_train_rf)
predictions_rfnorm = rfnorm_model.predict(X_test_rf)
print("Normalized Random Forest Classification Report (with upsampling):")
print(classification_report(y_test_rf, predictions_rfnorm))
print("Accuracy:", accuracy_score(y_test_rf, predictions_rfnorm))
update_results("Random Forest", "Normalized", y_test_rf, predictions_rfnorm)

###############################################################################
# Gradient Boosting with summarized_df #######################################
###############################################################################
X_train_rf, X_test_rf, y_train_rf, y_test_rf = apply_upsampling(
    dv.summarized_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White']
)
gbsum_model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=42)
gbsum_model.fit(X_train_rf, y_train_rf)
predictions_gbsum = gbsum_model.predict(X_test_rf)
print("Summarized Gradient Boosting Classification Report:")
print(classification_report(y_test_rf, predictions_gbsum))
print("Accuracy:", accuracy_score(y_test_rf, predictions_gbsum))
update_results("Gradient Boosting", "Summarized", y_test_rf, predictions_gbsum)
 # Save GBsum model
joblib.dump(gbsum_model, 'Models/gbsum_model.pkl')

###############################################################################
# Gradient Boosting with rate_change_df #######################################
###############################################################################
X_train_rf, X_test_rf, y_train_rf, y_test_rf = apply_upsampling(
    dv.rate_change_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White Rate Change']
)
gbrc_model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=42)
gbrc_model.fit(X_train_rf, y_train_rf)  # Reusing the RF train split
predictions_gbrc = gbrc_model.predict(X_test_rf)
print("Rate Change Gradient Boosting Classification Report:")
print(classification_report(y_test_rf, predictions_gbrc))
print("Accuracy:", accuracy_score(y_test_rf, predictions_gbrc))
update_results("Gradient Boosting", "Rate Change", y_test_rf, predictions_gbrc)

###############################################################################
# Gradient Boosting with normalized_df #######################################
###############################################################################
X_train_rf, X_test_rf, y_train_rf, y_test_rf = apply_upsampling(
    dv.normalized_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White Normalized']
)
gbnorm_model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=42)
gbnorm_model.fit(X_train_rf, y_train_rf)  # Reusing the RF train split
predictions_gbnorm = gbnorm_model.predict(X_test_rf)
print("Normalized Gradient Boosting Classification Report:")
print(classification_report(y_test_rf, predictions_gbnorm))
print("Accuracy:", accuracy_score(y_test_rf, predictions_gbnorm))
update_results("Gradient Boosting", "Normalized", y_test_rf, predictions_gbnorm)

###############################################################################
# SVM with summarized_df #####################################################
###############################################################################
X_train_knn, X_test_knn, y_train_knn, y_test_knn = apply_upsampling(
    dv.summarized_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White']
)
svmsum_model = SVC(C=1, kernel='linear')
svmsum_model.fit(X_train_knn, y_train_knn)
predictions_svmsum = svmsum_model.predict(X_test_knn)
print("Summarized SVM Classification Report:")
print(classification_report(y_test_knn, predictions_svmsum))
print("Accuracy:", accuracy_score(y_test_knn, predictions_svmsum))
update_results("SVM", "Summarized", y_test_knn, predictions_svmsum)

###############################################################################
# SVM with rate_change_df ####################################################
###############################################################################
X_train_knn, X_test_knn, y_train_knn, y_test_knn = apply_upsampling(
    dv.rate_change_df,
    'gentrification_index',
    ['gentrification_index', 'gentrification_category', 'Area', 'Year', 'Percentage White Rate Change']
)
svmrc_model = SVC(C=1, kernel='linear')
svmrc_model.fit(X_train_knn, y_train_knn)  # Reusing the KNN train split
predictions_svmrc = svmrc_model.predict(X_test_knn)
print("Rate Change SVM Classification Report:")
print(classification_report(y_test_knn, predictions_svmrc))
print("Accuracy:", accuracy_score(y_test_knn, predictions_svmrc))
update_results("SVM", "Rate Change", y_test_knn, predictions_svmrc)

###############################################################################
# SVM with normalized_df ######################################################
###############################################################################
svmnorm_model = SVC(C=1, kernel='linear')
svmnorm_model.fit(X_train_knn, y_train_knn)  # Reusing the KNN train split
predictions_svmnorm = svmnorm_model.predict(X_test_knn)
print("Normalized SVM Classification Report:")
print(classification_report(y_test_knn, predictions_svmnorm))
print("Accuracy:", accuracy_score(y_test_knn, predictions_svmnorm))
update_results("SVM", "Normalized", y_test_knn, predictions_svmnorm)
# Save SVMnorm model
joblib.dump(svmnorm_model, 'Models/svmnorm_model.pkl')

# Save results to a CSV file
results_df.to_csv('Models/results_df.csv', index=False)
"""
def perform_grid_search(model, param_grid, X_train, y_train, X_test, y_test):
    """"""Perform grid search and evaluate the model.""""""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return grid_search.best_params_, accuracy, report

# Parameters for grid search
param_grids = {
    LogisticRegression: {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']},
    KNeighborsClassifier: {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance']},
    RandomForestClassifier: {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
    GradientBoostingClassifier: {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
}

# DataFrames and their respective drops
dfs = {
    'summarized_df': dv.summarized_df,
    'rate_change_df': dv.rate_change_df,
    'normalized_df': dv.normalized_df
}
drop_cols = {
    'summarized_df': ['Area', 'Year', 'Percentage White'],
    'rate_change_df': ['Area', 'Year', 'Percentage White Rate Change'],
    'normalized_df': ['Area', 'Year', 'Percentage White Normalized']
}

# Evaluate each model on each dataframe
results = {}
for df_name, df in dfs.items():
    print(f"Processing {df_name}:")
    X_train, X_test, y_train, y_test = prepare_data(df, 'gentrification_index', drop_cols[df_name])
    X_train_up, y_train_up = apply_upsampling(X_train, y_train)
    
    for Model, params in param_grids.items():
        model = Model()
        best_params, acc_up, report_up = perform_grid_search(model, params, X_train_up, y_train_up, X_test, y_test)
        _, acc, report = perform_grid_search(model, params, X_train, y_train, X_test, y_test)
        
        print(f"{Model.__name__} on {df_name} - Upsampled: {acc_up}, Non-Upsampled: {acc}")
        print(f"Best parameters: {best_params}")
        print(f"Classification report (upsampled):\n{report_up}")
        print(f"Classification report (non-upsampled):\n{report}")
        results[(Model.__name__, df_name, 'upsampled')] = (acc_up, best_params)
        results[(Model.__name__, df_name, 'non-upsampled')] = (acc, best_params)
"""
