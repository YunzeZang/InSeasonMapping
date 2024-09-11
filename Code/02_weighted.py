
from utility_trainInSeasonRF import *
from utility_priorCM import *
from utility_prepareData import *
import pickle
import os
from ICS import ICS


def main():

    # parameters
    regionName = 'site_I'
    modelName = 'weighted'

    # Total number of positive and negative samples
    sampleNumber = 6000 
    year = 2021 # current year
    modelPath = './rf.pkl' # HSC's path, if available

    # The confusion matrix of trusted samples in the year prior to the current year
    # Here is 2020
    prioCM = getPrioCM(regionName) 

    # Load data
    trainingData = getTrainingSample(regionName, startYear=2016, endYear=year)
    trusted_p, trusted_n = getTrustedSample(regionName, year)
    unlabelData = getUnlabelData(regionName)
    testData = getTestSample(regionName, year)
    
    # Load the growing season
    list_DGS = getDayOfGrowingSeason(regionName)

    resultsList = []
    
    for i in list_DGS:

        # Acquire the feature index for the current time point
        featureIndex = np.arange(0,np.floor(i/10),dtype=int)
        # Construct the feature column names for the current time point
        currentFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] for num in featureIndex], dtype=str).flatten()

        X_train = trainingData[currentFeatures]
        y_train = trainingData[classProperty]

        X_trusted_p = trusted_p[currentFeatures]
        X_trusted_n = trusted_n[currentFeatures]

        X_unlabel = unlabelData[currentFeatures]

        X_test = testData[currentFeatures]
        y_test = testData[classProperty]

        # Load the HSC, if available. Otherwise, a HSC using random forest is trained
        if os.path.exists(modelPath):
            _, ext = os.path.splitext(modelPath)
            if ext.lower() == '.pkl':
                with open(modelPath, 'rb') as file:
                    initial_classifier = pickle.load(file)
            else:
                 # if your model is deep learning model, please read it here
                 # make sure the output of the model is probability
                raise ValueError("Please read your model")
        else:
            # If no model is available, train a random forest
            classifier_list = train_random_forest(X_train,y_train,times=1)
            initial_classifier = classifier_list[0]

        # Split trusted sample into two parts:
        # "est" part is used to estimate the weight and the RSP
        # "samp" part is used to resample the samples
        X_trusted_p_est, X_trusted_p_samp = train_test_split(X_trusted_p, test_size=0.5)
        X_trusted_n_est, X_trusted_n_samp = train_test_split(X_trusted_n, test_size=0.5)
        
        # Get the classification probability using the HSC
        y_predicted_p_est = np.array([prob[1] for prob in initial_classifier.predict_proba(X_trusted_p_est)])
        y_predicted_n_est = np.array([prob[1] for prob in initial_classifier.predict_proba(X_trusted_n_est)])

        y_predicted_p_samp = np.array([prob[1] for prob in initial_classifier.predict_proba(X_trusted_p_samp)])
        y_predicted_n_samp = np.array([prob[1] for prob in initial_classifier.predict_proba(X_trusted_n_samp)])

        y_predicted_c = np.array([prob[1] for prob in initial_classifier.predict_proba(X_unlabel)]) # classified samples

        # Add the probability as a column
        X_trusted_n_samp['prob'] = y_predicted_n_samp
        X_trusted_p_samp['prob'] = y_predicted_p_samp
        X_unlabel['prob'] = y_predicted_c

        # initialize sample weighting class ICS
        ics = ICS(prioCM = prioCM, sampleNumber=sampleNumber, binWidth=0.05)
        # estimate the RSP (resampling proportion) of trusted samples and classified samples
        ics.estimate(y_predicted_p_est, y_predicted_n_est, y_predicted_c)
        print(ics.RSP_t_p,ics.RSP_t_n,ics.RSP_c)
        # resample the samples from trusted samples and classified samples
        X_train_w, y_train_w = ics.weight(X_trusted_p_samp,X_trusted_n_samp,X_unlabel)


        # train the weighted classifier
        # if your model is deep learning model, please finetune it here
        classifier_list = train_random_forest(X_train_w[currentFeatures], y_train_w)
        
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