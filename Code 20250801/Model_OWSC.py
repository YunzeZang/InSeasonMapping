import os
from utility.utility_trainInSeasonRF import *
from utility.utility_priorCM import *
from utility.utility_prepareData import *
from utility.ICS import ICS
from utility.utility_trainDL import *
from torch.utils.data import DataLoader, TensorDataset
import pickle

def main(region_name, year=2021, yearLength=10, model_name='RF', method_name='OWSC', 
         device='cpu', inseasonT=None, sampleNumber=3000 * 2):
    """
    Main function for training crop classification models with different sampling strategies.
    
    Args:
        region_name (str): Name of the study region
        year (int): Target year for classification
        yearLength (int): Number of years to consider for historical data
        model_name (str): Model architecture ('RF', 'SVM', or deep learning models)
        method_name (str): Sampling strategy ('OWSC', 'EWSC')
        device (str): Device to run training on ('cpu' or 'cuda')
        inseasonT (int, optional): Specific day of growing season to process
        sampleNumber (int): Number of samples to use for training
    """
    # Determine model file extension based on model type
    model_fix = '.pth' if model_name in DL_model_list else '.pkl'

    # Get prior confusion matrix from previous year
    prioCM = getPrioCM(region_name, year=year, yearLength=yearLength)

    # Get growing season days for the region
    list_DGS = getDayOfGrowingSeason(region_name)

    # Load trusted and unlabeled data
    trusted_p, trusted_n = getTrustedSample(region_name, year=year, yearLength=yearLength)
    unlabelData = getUnlabelData(region_name, year)
    
    # Prepare all features and compute normalization parameters
    all_featureIndex = np.arange(0, np.floor(list_DGS[-1]/10), dtype=int)
    allFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] 
                          for num in all_featureIndex], dtype=str).flatten()
    testData = getTestSample(region_name, year)
    meanS2, stdS2 = getNormalizePara(inseasonInterpol(testData[allFeatures], featureNames), allFeatures)

    # Process specific day if provided, otherwise process all days
    if inseasonT is not None:
        list_DGS = [inseasonT]

    for i in list_DGS:
        # Set up model paths
        HSC_path = os.path.join(outputPath, model_name, 'HSC', f'model_{year}_{i}{model_fix}')
        model_dir = os.path.join(outputPath, model_name, method_name)
        os.makedirs(model_dir, exist_ok=True)
        weightedClassifier_path = os.path.join(model_dir, f'model_{year}_{i}{model_fix}')
    
        # Prepare features for current time point
        featureIndex = np.arange(0, np.floor(i/10), dtype=int)
        currentFeatures = np.array([[f + '_10day_' + str(num) for f in featureNames] 
                                  for num in featureIndex], dtype=str).flatten()

        # Process and split trusted samples
        X_trusted_p = inseasonInterpol(trusted_p[currentFeatures], featureNames)[currentFeatures]
        X_trusted_n = inseasonInterpol(trusted_n[currentFeatures], featureNames)[currentFeatures]
        X_unlabel_raw = inseasonInterpol(unlabelData[currentFeatures], featureNames)[currentFeatures]
        
        # Split trusted samples into estimation and sampling sets
        X_trusted_p_est, X_trusted_p_samp = train_test_split(X_trusted_p, test_size=0.5, random_state=random_seed)
        X_trusted_n_est, X_trusted_n_samp = train_test_split(X_trusted_n, test_size=0.5, random_state=random_seed)
        
        # Prepare copies of data for processing
        X_trusted_n_samp_copy = X_trusted_n_samp.copy()
        X_trusted_p_samp_copy = X_trusted_p_samp.copy()
        X_unlabel = X_unlabel_raw[currentFeatures].copy()
        X_unlabel_copy = X_unlabel_raw[currentFeatures].copy()

        # Load Historical Sample Classifier (HSC)
        if os.path.exists(HSC_path):
            _, ext = os.path.splitext(HSC_path)
            if ext.lower() == '.pth':
                # Prepare parameters for deep learning model
                modelParameter = {
                    'input_dim': feature_len, 
                    'num_classes': 2, 
                    'max_seq_len': int(X_unlabel.shape[1]/feature_len)
                }
                
                # Load model and prepare data tensors
                initial_classifier, _, scheduler, _ = loadModel(
                    model_name, modelParameter, checkPath=HSC_path, device=device
                )
                
                # Convert data to tensors and normalize
                X_trusted_p_est = torch.tensor(
                    np.reshape(X_trusted_p_est.values, (X_trusted_p_est.shape[0], -1, feature_len)),
                    dtype=torch.float32
                )
                X_trusted_n_est = torch.tensor(
                    np.reshape(X_trusted_n_est.values, (X_trusted_n_est.shape[0], -1, feature_len)),
                    dtype=torch.float32
                )
                X_trusted_p_samp = torch.tensor(
                    np.reshape(X_trusted_p_samp.values, (X_trusted_p_samp.shape[0], -1, feature_len)),
                    dtype=torch.float32
                )
                X_trusted_n_samp = torch.tensor(
                    np.reshape(X_trusted_n_samp.values, (X_trusted_n_samp.shape[0], -1, feature_len)),
                    dtype=torch.float32
                )
                X_unlabel = torch.tensor(
                    np.reshape(X_unlabel.values, (X_unlabel.shape[0], -1, feature_len)),
                    dtype=torch.float32
                )
                
                # Normalize data
                X_trusted_p_est = (X_trusted_p_est - meanS2) / stdS2
                X_trusted_n_est = (X_trusted_n_est - meanS2) / stdS2
                X_trusted_p_samp = (X_trusted_p_samp - meanS2) / stdS2
                X_trusted_n_samp = (X_trusted_n_samp - meanS2) / stdS2
                X_unlabel = (X_unlabel - meanS2) / stdS2
                
            elif ext.lower() == '.pkl':
                # Load machine learning model
                with open(HSC_path, 'rb') as f:
                    initial_classifier = pickle.load(f)
        else:
            raise ValueError("HSC model not found. Please check the file path.")
        
        # Get predictions from initial classifier
        y_predicted_p_est = np.array([prob[1] for prob in initial_classifier.predict_proba(X_trusted_p_est)])
        y_predicted_n_est = np.array([prob[1] for prob in initial_classifier.predict_proba(X_trusted_n_est)])
        y_predicted_c = np.array([prob[1] for prob in initial_classifier.predict_proba(X_unlabel)])

        # Add probabilities to dataframes
        X_trusted_n_samp_copy['prob'] = np.array([prob[1] for prob in initial_classifier.predict_proba(X_trusted_n_samp)])
        X_trusted_p_samp_copy['prob'] = np.array([prob[1] for prob in initial_classifier.predict_proba(X_trusted_p_samp)])
        X_unlabel_copy['prob'] = y_predicted_c

        # Weight samples based on selected method
        if method_name == 'OWSC':
            # Optimal Weighted Sample Classifier
            ics = ICS(prioCM=prioCM, sampleNumber=sampleNumber, binWidth=0.05)
            ics.estimate(y_predicted_p_est, y_predicted_n_est, y_predicted_c)
            print(f"Resampling proportions: Trusted positive={ics.RSP_t_p}, Trusted negative={ics.RSP_t_n}, Classified={ics.RSP_c}")
            X_train_w, y_train_w = ics.weight(X_trusted_p_samp_copy, X_trusted_n_samp_copy, X_unlabel_copy)
            X_train_w['label'] = y_train_w
            
        elif method_name == 'EWSC':
            # Equal Weighted Sample Classifier
            
            X_trusted_p_train = X_trusted_p[currentFeatures].sample(int(sampleNumber/4))
            X_trusted_n_train = X_trusted_n[currentFeatures].sample(int(sampleNumber/4))
            
            X_unlabel_p_train = X_unlabel_copy[X_unlabel_copy['prob'] >= 0.5][currentFeatures].sample(int(sampleNumber/4))
            X_unlabel_n_train = X_unlabel_copy[X_unlabel_copy['prob'] < 0.5][currentFeatures].sample(int(sampleNumber/4))
            
            # Assign labels and combine datasets
            X_trusted_p_train['label'] = 1
            X_trusted_n_train['label'] = 0
            X_unlabel_p_train['label'] = 1
            X_unlabel_n_train['label'] = 0
            
            X_train_w = pd.concat([X_trusted_p_train, X_trusted_n_train, 
                                 X_unlabel_p_train, X_unlabel_n_train])
            y_train_w = X_train_w['label']

        # Train the final classifier
        if model_name in DL_model_list:
            # Prepare data for deep learning
            X_train_w = X_train_w[currentFeatures].values
            X_train_w = np.reshape(X_train_w, (X_train_w.shape[0], -1, feature_len))
            X_train_w = torch.tensor((X_train_w - meanS2) / stdS2, dtype=torch.float32)
            y_train_w = torch.tensor(np.array(y_train_w), dtype=torch.long)
            
            # Create datasets and dataloaders
            all_dataset = TensorDataset(X_train_w, y_train_w)
            train_dataset, valid_dataset = random_split(
                all_dataset, 
                [int(len(all_dataset) * 0.7), 
                 len(all_dataset) - int(len(all_dataset) * 0.7)]
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

            # Initialize model and train
            model, optimizer, scheduler, criterion = loadModel(
                model_name, modelParameter=modelParameter, checkPath='', device=device
            )
            
            best_valid_loss = float('inf')
            patience_counter = 0

            for epoch in range(num_epochs):
                train_loss = train(model, train_loader, criterion, optimizer, device)
                valid_loss = validate(model, valid_loader, criterion, device)
                scheduler.step(valid_loss[0])
                
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Validation Loss: {valid_loss[0]:.4f}")
                print(f"Validation Acc: {valid_loss[1]:.4f}\n")

                # Early stopping check
                if valid_loss[0] < best_valid_loss:
                    best_valid_loss = valid_loss[0]
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                    }, weightedClassifier_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break

        elif model_name in ML_model_list:
            # Train machine learning model
            if model_name == 'RF':
                classifier_list, best_index = train_random_forest(X_train_w[currentFeatures], y_train_w)
            elif model_name == 'SVM':
                classifier_list, best_index = train_SVM(X_train_w[currentFeatures], y_train_w)
            
            # Save trained model
            with open(weightedClassifier_path, 'wb') as f:
                pickle.dump(classifier_list[best_index], f)

# Configuration for running the script
region_name = 'CS_C'
year = 2021
method_nameList = ['HSC', 'TSC', 'CSC']
method_name = 'OWSC'
model_name = 'RF'
device = torch.device("cuda:0")
basePath = r'F:\InseasonMappingIntegrateRotation\Results\Deep Model\\'
outputPath = 'trained_model'

# Run the main function
main(region_name=region_name, model_name=model_name, method_name=method_name, 
     year=2021, device=device, inseasonT=140)