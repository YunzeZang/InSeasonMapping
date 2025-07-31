import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utility.utility_prepareData import *
from utility.utility_trainInSeasonRF import *
from utility.utility_trainDL import *
import pickle

def main(regionName, model_name, method_name, year, device, inseasonT=None):
    """
    Main function for running crop classification models.
    
    Args:
        regionName (str): Name of the study region
        model_name (str): Name of the model to use (e.g., 'Transformer', 'SVM')
        method_name (str): Name of the classification method (e.g., 'HSC', 'TSC')
        year (int): Target year for classification
        device (torch.device): Device to run the model on (default: CPU)
        inseasonT (int): Specific day of growing season to process (optional)
    """
    # Configuration parameters
    batch_size = 512
    F1_type = 'binary'  # Evaluation metric type
    
    # Get days of growing season for the region
    list_DGS = getDayOfGrowingSeason(regionName)

    # Prepare test data
    featureIndex = np.arange(0, np.floor(list_DGS[-1]/10), dtype=int)
    currentFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] 
                               for num in featureIndex], dtype=str).flatten()
    testData = getTestSample(regionName, year)
    y_test = testData[classProperty].values
    
    # Compute normalization parameters
    all_featureIndex = np.arange(0, np.floor(list_DGS[-1]/10), dtype=int)
    allFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] 
                           for num in all_featureIndex], dtype=str).flatten()
    meanS2, stdS2 = getNormalizePara(inseasonInterpol(testData[allFeatures], featureNames), allFeatures)

    results_list = []

    # Process specific day if provided, otherwise process all days
    if inseasonT is not None:
        list_DGS = [inseasonT]

    for i in list_DGS:
        # Prepare features for current day of growing season
        featureIndex = np.arange(0, np.floor(i/10), dtype=int)
        currentFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] 
                                   for num in featureIndex], dtype=str).flatten()
        
        # Identify cloudy pixels
        cloudTag = testData['B2_10day_' + str(featureIndex[-1])].isna().values
        X_test = inseasonInterpol(testData[currentFeatures], featureNames)[currentFeatures]

        # Deep Learning model processing
        if model_name in DL_model_list:
            model_path = os.path.join(outputPath, model_name, method_name, f'model_{year}_{i}.pth')

            # Prepare input tensor
            X_test = np.reshape(X_test.values, (X_test.shape[0], -1, 15))
            X_test = torch.tensor((X_test - meanS2) / stdS2, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)

            # Create dataloader for batch processing
            all_dataset = TensorDataset(X_test)
            valid_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

            # Load and run model
            modelParameter = {'input_dim': feature_len, 'num_classes': 2, 'max_seq_len': X_test.shape[1]}
            model, _, _, _ = loadModel(model_name, modelParameter=modelParameter, 
                                      checkPath=model_path, device=device)
            y_predicted = np.array(DL_predict(model, valid_loader, device) >= 0.5, dtype=int)
            
        # Machine Learning model processing
        elif model_name in ML_model_list:
            model_path = os.path.join(outputPath, model_name, method_name, f'model_{year}_{i}.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            y_predicted = model.predict(X_test)

        # Handle cloudy pixels by using previous results
        os.makedirs(os.path.join(outputPath, 'classification_result', model_name, method_name), exist_ok=True)

        previous_result_path = os.path.join(outputPath, 'classification_result', 
                                          model_name, method_name, f'result_temporal_{i-10}.csv')
        
        if os.path.exists(previous_result_path):
            previous_result = pd.read_csv(previous_result_path)['Result']
            y_predicted[cloudTag] = previous_result[cloudTag]

        # Save current results
        current_result_path = os.path.join(outputPath, 'classification_result', 
                                         model_name, method_name, f'result_temporal_{i}.csv')
        pd.DataFrame({'Result': y_predicted}).to_csv(current_result_path)
        
        # Calculate evaluation metrics
        CM = confusion_matrix(y_test, y_predicted)
        F1_Score = f1_score(y_test, y_predicted, average=F1_type)
        Precision = precision_score(y_test, y_predicted, average=F1_type)
        Recall = recall_score(y_test, y_predicted, average=F1_type)

        # Store results
        results_list.append({
            'F1Score': F1_Score,
            'Precision': Precision,
            'Recall': Recall,
            'CM': CM
        })

        # Prepare evaluation dataframe
        F1_Score_List = [result['F1Score'] for result in results_list]
        precision_List = [result['Precision'] for result in results_list]
        recall_List = [result['Recall'] for result in results_list]
        CM_List = [result['CM'] for result in results_list]

        accuracyEvaluation = pd.DataFrame({
            'F1_Score': F1_Score_List,
            'Precision': precision_List,
            'Recall': recall_List,
            'CM': CM_List,
        })
        print('Accuracy Evaluation:', accuracyEvaluation)

        # Save evaluation results
        os.makedirs(os.path.join(outputPath, 'evaluation_result', model_name, method_name), exist_ok=True)
        out_evaluation_path = os.path.join(outputPath, 'evaluation_result', 
                                         model_name, method_name, f'Evaluation_{regionName}_{year}.csv')
        accuracyEvaluation.to_csv(out_evaluation_path)

# Configuration for running the script
region_name = 'CS_C'
year = 2021
method_nameList = ['HSC', 'TSC', 'CSC']  # Available classification methods
method_name = 'OWSC'  # Selected method
model_name = 'RF'  # Model to use
device = torch.device("cuda:0")  # Use GPU if available
outputPath = 'trained_model'  # Output directory
inseasonT = 140  # Specific day of growing season to process

# Run the main function
main(region_name, model_name, method_name, year, device=device, inseasonT=inseasonT)