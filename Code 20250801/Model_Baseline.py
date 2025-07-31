import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utility.utility_prepareData import *
from utility.utility_trainInSeasonRF import *
from utility.utility_trainDL import *
import pickle

def main(region_name, model_name, method_name, year, device, inseasonT=None, sampleNumber=3000):
    """
    Main training function for crop classification models.
    
    Args:
        region_name (str): Name of the study region
        model_name (str): Name of the model architecture
        method_name (str): Training method ('CSC', 'HSC', or 'TSC')
        year (int): Target year for classification
        device (torch.device): Device to run training on
        inseasonT (int, optional): Specific day of growing season to process
        sampleNumber (int): Number of samples to use for training
    """
    # Configuration parameters
    batch_size = 512
    num_epochs = 500
    patience = 50
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('inf')
    patience_counter = 0

    # Load training data based on method type
    if method_name == 'CSC':  # Current-year sample classifier
        trainingData = getTrainingSample(region_name, startYear=year, 
                                       endYear=year + 1, sampleNumber=sampleNumber)
    elif method_name == 'HSC':  # Historical sample classifier
        startYear = 2017 if region_name == 'Qinghai' else 2016
        trainingData = getTrainingSample(region_name, startYear=startYear, 
                                       endYear=year, sampleNumber=sampleNumber)
    elif method_name == 'TSC':  # Trusted sample classifier
        trusted_p, trusted_n = getTrustedSample(region_name, year=year, yearLength=10)
        trusted_p['Landcover'] = 1
        trusted_n['Landcover'] = 0
        trainingData = pd.concat([trusted_p.sample(sampleNumber), 
                                 trusted_n.sample(sampleNumber)])
    else:
        raise ValueError(f"Invalid method name: {method_name}. Expected 'CSC', 'HSC', or 'TSC'.")

    # Get growing season days and prepare features
    list_DGS = getDayOfGrowingSeason(region_name)
    featureIndex = np.arange(0, np.floor(list_DGS[-1]/10), dtype=int)
    currentFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] 
                               for num in featureIndex], dtype=str).flatten()
    
    # Load test data for normalization
    testData = getTestSample(region_name, year)
    trainingLabel = trainingData[classProperty].values
    
    # Prepare all features and compute normalization parameters
    all_featureIndex = np.arange(0, np.floor(list_DGS[-1]/10), dtype=int)
    allFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] 
                          for num in all_featureIndex], dtype=str).flatten()
    trainingData = inseasonInterpol(trainingData[allFeatures], featureNames)
    S2_mean, S2_std = getNormalizePara(inseasonInterpol(testData[allFeatures], featureNames), allFeatures)

    # Process specific day if provided, otherwise process all days
    if inseasonT is not None:
        list_DGS = [inseasonT]

    for i in list_DGS:
        # Prepare features for current day
        featureIndex = np.arange(0, np.floor(i/10), dtype=int)
        currentFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] 
                                  for num in featureIndex], dtype=str).flatten()

        # Create model directory if it doesn't exist
        model_dir = os.path.join(outputPath, model_name, method_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Deep Learning model training
        if model_name in DL_model_list:
            # Prepare training data tensors
            X_train = trainingData[currentFeatures].values
            X_train = np.reshape(X_train, (X_train.shape[0], -1, feature_len))
            X_train = (X_train - S2_mean) / S2_std
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(trainingLabel, dtype=torch.long)

            # Create datasets and dataloaders
            all_dataset = TensorDataset(X_train, y_train)
            train_dataset, valid_dataset = random_split(
                all_dataset, 
                [int(len(all_dataset) * 0.7), 
                 len(all_dataset) - int(len(all_dataset) * 0.7)]
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

            # Initialize model
            model_path = os.path.join(model_dir, f'model_{year}_{i}.pth')
            modelParameter = {
                'input_dim': feature_len, 
                'num_classes': 2, 
                'max_seq_len': X_train.shape[1]
            }
            model, optimizer, scheduler, criterion = loadModel(
                model_name, 
                modelParameter=modelParameter, 
                checkPath='', 
                device=device
            )

            # Training loop
            for epoch in range(num_epochs):
                print(f'Epoch {epoch + 1}/{num_epochs}')
                train_loss = train(model, train_loader, criterion, optimizer, device)
                valid_loss = validate(model, valid_loader, criterion, device)
                
                scheduler.step(valid_loss[0])
                
                print(f"Train Loss: {train_loss:.4f} "
                      f"Validation Loss: {valid_loss[0]:.4f}\n"
                      f"Validation Acc: {valid_loss[1]:.4f}\n")

                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)

                # Save best model and check for early stopping
                if valid_loss[0] < best_valid_loss:
                    best_valid_loss = valid_loss[0]
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss_list,
                        'vaild_loss': valid_loss_list,
                        'loss': train_loss,
                    }, model_path)
                    print(f"Model {model_name} saved at {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break

        # Machine Learning model training
        elif model_name in ML_model_list:
            model_path = os.path.join(model_dir, f'model_{year}_{i}.pkl')
            X_train = trainingData[currentFeatures].values
            y_train = trainingLabel

            # Train Random Forest or SVM
            if model_name == 'RF':
                classifier_list, best_index = train_random_forest(X_train, y_train)
            elif model_name == 'SVM':
                classifier_list, best_index = train_SVM(X_train, y_train)
            
            # Save trained model
            with open(model_path, 'wb') as f:
                pickle.dump(classifier_list[best_index], f)
            print(f"Model {model_name} saved at {model_path}")
            
        else:
            print('Invalid model name! Supported models:')
            print('Transformer, DCM, TempCNN, RF, SVM')

# Configuration for running the script
region_name = 'CS_C'
year = 2021
method_nameList = ['HSC', 'TSC', 'CSC']  # Available training methods
method_name = 'HSC'  # Selected method
model_name = 'RF'  # Model architecture
device = torch.device("cuda:0")  # Use GPU if available
outputPath = 'trained_model'  # Output directory

# Run the main training function
main(region_name=region_name, model_name=model_name, method_name=method_name, 
     year=2021, device=device, inseasonT=140)