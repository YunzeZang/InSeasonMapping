from utility_trainInSeasonRF import *
from utility_priorCM import *
from utility_prepareData import *
import pandas as pd
import numpy as np



def main():

    # Define region name and model name
    regionName = 'site_I'
    modelName = 'CSC'

    # Set sample number and year
    sampleNumber = 3000
    year = 2021

    # Acquire training data based on the model type
    if modelName == 'CSC':  # Current-year sample classifier
        trainingData = getTrainingSample(regionName, startYear=year, endYear=year + 1)
    elif modelName == 'HSC':  # Historical sample classifier
        trainingData = getTrainingSample(regionName, startYear=2016, endYear=year)
    elif modelName == 'TSC':  # Trusted sample classifier
        trusted_p, trusted_n = getTrustedSample(regionName, startYear=year, endYear=year + 1)
        trainingData = pd.concat([trusted_p.sample(sampleNumber), trusted_n.sample(sampleNumber)])
    else:
        raise ValueError(f"Invalid model name: {modelName}. Expected one of 'CSC', 'HSC', or 'TSC'.")
    
    # Acquire test data
    testData = getTestSample(regionName, year)

    # Acquire the list of "days of growing season"
    list_DGS = getDayOfGrowingSeason(regionName)

    # Initialize the results list
    resultsList = []

    # Iterate over each day of the growing season
    for i in list_DGS:
        # Acquire the feature index for the current time point
        featureIndex = np.arange(0,np.floor(i/10),dtype=int)
        # Construct the feature column names for the current time point
        currentFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] for num in featureIndex], dtype=str).flatten()

        X_train = trainingData[currentFeatures]
        y_train = trainingData[classProperty]

        X_test = testData[currentFeatures]
        y_test = testData[classProperty]

        # Train the random forest classifiers
        classifier_list = train_random_forest(X_train, y_train)

        # Evaluate the classifier performance
        result = evaluate_classification(X_test, y_test, classifier_list)

        print(result)

        resultsList.append(result)

    print('multiClassificationResults:', resultsList)

    F1_Score_List = [result['F1Score'] for result in resultsList]
    precision_List = [result['Precision'] for result in resultsList]
    recall_List = [result['Recall'] for result in resultsList]

    accuracyEvaluation = pd.DataFrame({
        'F1_Score': F1_Score_List,
        'Precision': precision_List,
        'Recall': recall_List
    })

    print(accuracyEvaluation)

    accuracyEvaluation.to_csv(f'results_{modelName}_{regionName}_{year}.csv')

main()