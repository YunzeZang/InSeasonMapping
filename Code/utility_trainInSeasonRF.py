import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


def getDayOfGrowingSeason(regionName):
    if regionName=='site_I':
        return np.arange(10,230,10)


def train_random_forest(X_train, y_train, n_estimators = 100, times = 10):

    models = []
    for i in range(times):
        initial_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=i, oob_score=True)
        initial_classifier.fit(X_train, y_train)
        models.append(initial_classifier)
    return models

def evaluate_classification(X_test, y_test, models):
   
    multiEvaluationResults = []

    for model in models:
        y_predicted = model.predict(X_test)

        CM = confusion_matrix(y_test, y_predicted)
        F1_Score = f1_score(y_test, y_predicted)
        Precision = precision_score(y_test, y_predicted)
        Recall = recall_score(y_test, y_predicted)
        
        multiEvaluationResults.append({
            'F1Score': F1_Score,
            'Precision': Precision,
            'Recall': Recall,
            'CM': CM,
            'initial_classifier': model,
        })

    F1_Score_List = np.array([result['F1Score'] for result in multiEvaluationResults])
    Precision_List = np.array([result['Precision'] for result in multiEvaluationResults])
    Recall_List = np.array([result['Recall'] for result in multiEvaluationResults])
    

    meanF1 = F1_Score.mean()
    diff = np.absolute(F1_Score_List - meanF1)
    minDif = diff.min()
    minDifIndex = np.where(diff == minDif)

    return {
        'F1Score': F1_Score_List.mean(),
        'Precision': Precision_List.mean(),
        'Recall': Recall_List.mean(),
        'CM': multiEvaluationResults[int(minDifIndex[0][0])]['CM'],
        'initial_classifier': multiEvaluationResults[int(minDifIndex[0][0])]['initial_classifier']
    }


classProperty = 'Landcover'
featureNames = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','EVI','GCVI','LSWI','MNDWI']