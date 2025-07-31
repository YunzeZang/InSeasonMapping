import numpy as np
import pandas as pd
from utility.utility_priorCM import *
from utility.utility_trainInSeasonRF import *

def getTrainingSample(regionName, startYear, endYear, sampleNumber=3000, randomSeed=2025):
    """
    Acquires training data for a specified region and time range.
    
    Args:
        regionName (str): Name of the region.
        startYear (int): Starting year for data collection.
        endYear (int): Ending year for data collection.
        sampleNumber (int): Number of samples to include.
        
    Returns:
        pd.DataFrame: Training data DataFrame.
    """
    trainingData = pd.DataFrame({})
    for year in range(startYear, endYear):
        data = pd.read_csv(f'{trainingPath}Sample_training_{regionName}_{year}.csv')
        trainingData = pd.concat([trainingData, data])

    # Sample equal number of positive and negative cases
    positive = trainingData[trainingData['Landcover'] == 1].sample(sampleNumber, random_state=randomSeed)
    negative = trainingData[trainingData['Landcover'] == 0].sample(sampleNumber, random_state=randomSeed)
    return pd.concat([positive, negative]).replace(-9999, np.nan).reset_index(drop=True)

def getMultiClassTrainingSample(regionName, startYear, endYear, cropType = -1, binaryTag = True, sampleNumber=3000):
    """
    Acquires training data for a specified region and time range.
    
    Args:
        regionName (str): Name of the region.
        startYear (int): Starting year for data collection.
        endYear (int): Ending year for data collection.
        sampleNumber (int): Number of samples to include.
        
    Returns:
        pd.DataFrame: Training data DataFrame.
    """
    trainingData = pd.DataFrame({})
    for year in range(startYear, endYear):
        data = pd.read_csv(f'{trainingPath}Samples_MultiClass_training_{year}_{regionName}.csv')
        trainingData = pd.concat([trainingData, data])
    
    if binaryTag:
        # Convert multiclass to binary classification
        positive = trainingData[trainingData['Landcover'] == cropType].sample(sampleNumber, random_state=2025)
        negative = trainingData[trainingData['Landcover'] != cropType].sample(sampleNumber, random_state=2025)
        positive['Landcover'] = 1
        negative['Landcover'] = 0
        return pd.concat([positive, negative]).replace(-9999, np.nan).reset_index(drop=True)
    else:
        # Return multiclass samples directly
        return trainingData.sample(sampleNumber*4, random_state=2025).replace(-9999, np.nan).reset_index(drop=True)

def getHistoricalSample(regionName, startYear, endYear,cropTag=-1, sampleNumber=3000):
    """
    Acquires training data for a specified region and time range.
    
    Args:
        regionName (str): Name of the region.
        startYear (int): Starting year for data collection.
        endYear (int): Ending year for data collection.
        sampleNumber (int): Number of samples to include.
        
    Returns:
        pd.DataFrame: Training data DataFrame.
    """
    trainingData = pd.DataFrame({})
    if cropTag == -1:
        # Binary classification case
        for year in range(startYear, endYear):
            data = pd.read_csv(f'{trainingPath}Sample_training_{regionName}_{year}.csv')
            trainingData = pd.concat([trainingData, data])
        # Return all positive and negative samples separately
        positive = trainingData[trainingData['Landcover'] == 1]
        negative = trainingData[trainingData['Landcover'] == 0]
        return positive.replace(-9999, np.nan).reset_index(drop=True), negative.replace(-9999, np.nan).reset_index(drop=True)
    else:
        # Multiclass classification case
        for year in range(startYear, endYear):
            data = pd.read_csv(f'{trainingPath}Samples_MultiClass_training_{year}_{regionName}.csv')
            trainingData = pd.concat([trainingData, data])

        if cropTag == 0:
            # Special case for cropTag 0
            positive = trainingData[~trainingData['Landcover'].isin([1, 2, 3])].sample(sampleNumber, random_state=2025)
            negative = trainingData[trainingData['Landcover'].isin([1, 2, 3])].sample(sampleNumber, random_state=2025)
            positive['Landcover'] = 1
            negative['Landcover'] = 0
        else:
            # Normal multiclass case
            positive = trainingData[trainingData['Landcover'] == cropTag]
            negative = trainingData[trainingData['Landcover'] != cropTag]
            positive['Landcover'] = 1
            negative['Landcover'] = 0
        return positive.replace(-9999, np.nan).reset_index(drop=True), negative.replace(-9999, np.nan).reset_index(drop=True)

def getUnlabelData(regionName, year=2021):
    """
    Retrieves unlabelled data for a specific region and year.
    
    Args:
        regionName (str): Name of the region.
        year (int): Year of the data.
        
    Returns:
        pd.DataFrame: Unlabelled data DataFrame.
    """
    path = f'{unlabelPath}Sample_unlabel_{regionName}_{year}.csv'
    return pd.read_csv(path).replace(-9999, np.nan)

def getTestSample(regionName, year=2021, multiTag=False):
    """
    Gets validation data for a given region and year.
    
    Args:
        regionName (str): Name of the region.
        year (int): Year of the data.
        multiTag (bool): Flag for multiclass data.
        
    Returns:
        pd.DataFrame: Validation data DataFrame.
    """
    if multiTag == True:
        currentTrainingPath = f'{validationPath}Samples_MultiClass_validation_{year}_{regionName}.csv'
        return pd.read_csv(currentTrainingPath).replace(-9999, np.nan)
    if multiTag == False:
        currentTrainingPath = f'{validationPath}Sample_validation_{regionName}_{year}.csv'
        return pd.read_csv(currentTrainingPath).replace(-9999, np.nan)

def selectDF(data, num):
    """
    Selects a specified number of rows from a DataFrame.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
        num (int): Number of rows to select.
        
    Returns:
        pd.DataFrame: Selected rows.
    """
    if data.shape[0] == 0:
        return data
    return data.sample(num, replace=True)

def getTrustedSample(regionName, year, cropTag=-1, yearLength=10, N = 100000):
    """
    Retrieves rotation fusion data for a specific region and year.
    
    Args:
        regionName (str): Name of the region.
        year (int): Year of the data.
        cropTag (int): Crop type identifier.
        yearLength (int): Number of years to consider.
        N (int): Total number of samples to generate.
        
    Returns:
        tuple: Two DataFrames representing positive and negative samples.
    """
    prop = get_rotation_Prop(regionName,yearLength,year,cropTag=cropTag)

    if cropTag != -1:
        # Load multiclass rotation data
        trusted_p_alter = selectDF(pd.read_csv(f'{trustedSamplePath}MultiClassSample_RotationP_{regionName}_{cropTag}_{year}.csv'), int(N * prop[1, 1]))
        trusted_p_mono = selectDF(pd.read_csv(f'{trustedSamplePath}MultiClassSample_RotationP_{regionName}_{cropTag}_mono_{year}.csv'), int(N * prop[1, 0]))
        trusted_n_alter = selectDF(pd.read_csv(f'{trustedSamplePath}MultiClassSample_RotationN_{regionName}_{cropTag}_{year}.csv'), int(N * prop[0, 1]))
        trusted_n_mono = selectDF(pd.read_csv(f'{trustedSamplePath}MultiClassSample_RotationN_{regionName}_{cropTag}_mono_{year}.csv'), int(N * prop[0, 0]))
    else: 
        if yearLength==10:
            # Load 10-year rotation data
            trusted_p_alter = selectDF(pd.read_csv(f'{trustedSamplePath}Sample_RotationP_{regionName}_{year}.csv'), int(N * prop[1, 1]))
            trusted_p_mono = selectDF(pd.read_csv(f'{trustedSamplePath}Sample_RotationP_mono_{regionName}_{year}.csv'), int(N * prop[1, 0]))
            trusted_n_alter = selectDF(pd.read_csv(f'{trustedSamplePath}Sample_RotationN_{regionName}_{year}.csv'), int(N * prop[0, 1]))
            trusted_n_mono = selectDF(pd.read_csv(f'{trustedSamplePath}Sample_RotationN_mono_{regionName}_{year}.csv'), int(N * prop[0, 0]))
        if yearLength<10:
            # Load multi-year rotation data
            yearLength = yearLength-1
            trusted_p_alter = selectDF(pd.read_csv(f'{trustedSamplePath}\MultiYearRotation\MultiYearRotation_{regionName}_rotationP_{yearLength}.csv'), int(N * prop[1, 1]))
            trusted_p_mono = selectDF(pd.read_csv(f'{trustedSamplePath}\MultiYearRotation\MultiYearRotation_mono_{regionName}_rotationP_{yearLength}.csv'), int(N * prop[1, 0]))
            trusted_n_alter = selectDF(pd.read_csv(f'{trustedSamplePath}\MultiYearRotation\MultiYearRotation_{regionName}_rotationN_{yearLength}.csv'), int(N * prop[0, 1]))
            trusted_n_mono = selectDF(pd.read_csv(f'{trustedSamplePath}\MultiYearRotation\MultiYearRotation_mono_{regionName}_rotationN_{yearLength}.csv'), int(N * prop[0, 0]))
    # Combine altered and mono rotation samples
    trusted_p = pd.concat([trusted_p_alter, trusted_p_mono])
    trusted_n = pd.concat([trusted_n_alter, trusted_n_mono])

    return trusted_p.replace(-9999, np.nan).reset_index(drop=True), trusted_n.replace(-9999, np.nan).reset_index(drop=True)

def getNormalizePara(trainingData,temporalFeature):
    """
    Calculates normalization parameters (mean and std) for temporal features.
    
    Args:
        trainingData (pd.DataFrame): Training data containing temporal features.
        temporalFeature (list): List of temporal feature names.
        
    Returns:
        tuple: Mean and standard deviation arrays for normalization.
    """
    subTrainingData = trainingData[temporalFeature]
    timeSeriesLen = int(len(temporalFeature) / len(featureNames))
    subTrainingData = np.array(subTrainingData).reshape((-1, timeSeriesLen, len(featureNames)))

    meanS2 = np.mean(subTrainingData, axis=(0,1), keepdims=True)
    stdS2 = np.std(subTrainingData, axis=(0,1), keepdims=True)
    return meanS2,stdS2

def inseasonInterpol(data,featureNames):
    """
    Performs interpolation on time series data with missing values.
    
    Args:
        data (pd.DataFrame): Input data with missing values.
        featureNames (list): List of base feature names.
        
    Returns:
        pd.DataFrame: Interpolated data with missing values filled.
    """
    interpolData = pd.DataFrame({})
    for feature in featureNames:
        # Process each feature's time series separately
        selected_cols = [s for s in data.columns if s.startswith(feature+'_')]
        currentFeatureData = data[selected_cols]
        currentFeatureData_np = np.copy(currentFeatureData)
        
        # Interpolate missing values for each sample
        for i in range(currentFeatureData.shape[0]):
            if np.isnan(currentFeatureData_np[i, :]).all():
                currentFeatureData_np[i, :] = np.where(np.isnan(currentFeatureData_np[i, :]), -9999, currentFeatureData_np[i, :])
            currentFeatureData_np[i, :] = np.interp(np.arange(currentFeatureData_np.shape[1]), 
                                                  np.where(~np.isnan(currentFeatureData_np[i, :]))[0], 
                                                  currentFeatureData_np[i, ~np.isnan(currentFeatureData_np[i, :])])
        interpolData = pd.concat([interpolData, pd.DataFrame(currentFeatureData_np,columns=selected_cols)], axis=1)
    
    # Handle rows with all -9999 values by replacing with feature means
    all_neg_rows = (interpolData == -9999).all(axis=1)
    mean_values = interpolData[~all_neg_rows].mean()
    mean_values.fillna(0, inplace=True)
    interpolData.loc[all_neg_rows] = np.array(mean_values)

    return interpolData

# Define file paths for data storage
rootPath = r""
trainingPath = rootPath+"training\\"  # Path for training data
validationPath = rootPath+"validation\\"  # Path for validation data
trustedSamplePath = rootPath+"rotation\\"  # Path for rotation data
unlabelPath = rootPath+"unlabel\\"  # Path for unlabeled data