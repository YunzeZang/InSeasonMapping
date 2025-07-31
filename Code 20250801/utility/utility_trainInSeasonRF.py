import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

def getDayOfGrowingSeason(regionName):
    """
    Returns the day intervals for the growing season based on region.
    
    Args:
        regionName (str): Name of the region
        
    Returns:
        np.array: Array of days representing the growing season intervals
    """
    if regionName == 'CS_C':
        return np.arange(10, 230, 10)
    if regionName == 'MSSP':
        return np.arange(10, 190, 10)
    if regionName == 'ND':
        return np.arange(10, 190, 10)
    if regionName == 'CBE':
        return np.arange(10, 230, 10)
    if regionName == 'Qinghai':
        return np.arange(10, 190, 10)
    if regionName == 'NE':
        return np.arange(10, 150, 10)

def train_random_forest(X_train, y_train, n_estimators=100, times=10):
    """
    Trains multiple Random Forest classifiers and selects the best one.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        times: Number of models to train
        
    Returns:
        tuple: (list of trained models, index of best model)
    """
    models = []
    OOB_list = []
    
    for i in range(times):
        # Initialize and train Random Forest with different random states
        initial_classifier = RandomForestClassifier(n_estimators=n_estimators, 
                                                  random_state=i, 
                                                  oob_score=True)
        initial_classifier.fit(X_train, y_train)
        models.append(initial_classifier)
        OOB_list.append(initial_classifier.oob_score_)
    
    # Select model with highest OOB score
    meanOOB = np.array(OOB_list).max()
    diff = np.absolute(np.array(OOB_list) - meanOOB)
    minDif = diff.min()
    minDifIndex = np.where(diff == minDif)

    return models, minDifIndex[0][0]

def train_SVM(X_train, y_train):
    """
    Trains an SVM classifier using grid search for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        kernel: Kernel type for SVM
        C: Regularization parameter
        gamma: Kernel coefficient
        
    Returns:
        list: Contains the best SVM classifier found during grid search
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }

    model = SVC(probability=True)
    print('Begin searching...')
    
    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, 
                              scoring='accuracy', verbose=2, n_jobs=4)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

    return [grid_search.best_estimator_], 0

def convert_results(resultsList):
    """
    Converts evaluation results from list of dictionaries to DataFrame.
    
    Args:
        resultsList: List of evaluation result dictionaries
        
    Returns:
        pd.DataFrame: Formatted evaluation metrics
    """
    # Extract metrics from results list
    F1_Score_List = [result['F1Score'] for result in resultsList]
    F1_Score_std_List = [result['F1Score_std'] for result in resultsList]
    precision_List = [result['Precision'] for result in resultsList]
    precision_std_List = [result['Precision_std'] for result in resultsList]
    recall_List = [result['Recall'] for result in resultsList]
    recall_std_List = [result['Recall_std'] for result in resultsList]
    CM_List = [result['CM'] for result in resultsList]

    # Create formatted DataFrame
    accuracyEvaluation = pd.DataFrame({
        'F1_Score': F1_Score_List,
        'Precision': precision_List,
        'Recall': recall_List,
        'CM': CM_List,
        'F1_Score_std': F1_Score_std_List,
        'Precision_std': precision_std_List,
        'Recall_std': recall_std_List,
    })
    
    return accuracyEvaluation

# Configuration constants
classProperty = 'Landcover'  # Target variable name
featureNames = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 
                'NDVI', 'EVI', 'GCVI', 'LSWI', 'MNDWI']  # Feature names
feature_len = len(featureNames)  # Number of features