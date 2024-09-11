
import numpy as np
import pandas as pd
from utility_priorCM import *

def getTrainingSample(regionName, startYear, endYear, sampleNumber=3000):
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
        data = pd.read_csv(f'{trainingPath}Training_{regionName}_{year}.csv')
        trainingData = pd.concat([trainingData, data])

    # Sample equal number of positive and negative cases
    positive = trainingData[trainingData['Landcover'] == 1].sample(sampleNumber, random_state=np.random.default_rng(seed=0))
    negative = trainingData[trainingData['Landcover'] == 0].sample(sampleNumber, random_state=np.random.default_rng(seed=0))
    return pd.concat([positive, negative])

def getUnlabelData(regionName, year=2021):
    """
    Retrieves unlabelled data for a specific region and year.
    
    Args:
        regionName (str): Name of the region.
        year (int): Year of the data.
        
    Returns:
        pd.DataFrame: Unlabelled data DataFrame.
    """
    path = f'{unlabelPath}Unlabel_{regionName}_{year}.csv'
    return pd.read_csv(path)

def getTestSample(regionName, year=2021):
    """
    Gets validation data for a given region and year.
    
    Args:
        regionName (str): Name of the region.
        year (int): Year of the data.
        
    Returns:
        pd.DataFrame: Validation data DataFrame.
    """
    currentTrainingPath = f'{validationPath}Validation_{regionName}_{year}.csv'
    return pd.read_csv(currentTrainingPath)

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

def getTrustedSample(regionName, year):
    """
    Retrieves rotation fusion data for a specific region and year.
    
    Args:
        regionName (str): Name of the region.
        year (int): Year of the data.
        
    Returns:
        tuple: Two DataFrames representing positive and negative samples.
    """
    N = 100000
    prop = get_rotation_Prop(regionName)

    # Load and sample positive and negative data
    trusted_p_alter = selectDF(pd.read_csv(f'{trustedSamplePath}Trusted_alter_positive_{regionName}_{year}.csv'), int(N * prop[1, 1]))
    trusted_p_mono = selectDF(pd.read_csv(f'{trustedSamplePath}Trusted_mono_positive_{regionName}_{year}.csv'), int(N * prop[1, 0]))
    trusted_n_alter = selectDF(pd.read_csv(f'{trustedSamplePath}Trusted_alter_negative_{regionName}_{year}.csv'), int(N * prop[0, 1]))
    trusted_n_mono = selectDF(pd.read_csv(f'{trustedSamplePath}Trusted_mono_negative_{regionName}_{year}.csv'), int(N * prop[0, 0]))

    trusted_p = pd.concat([trusted_p_alter, trusted_p_mono])
    trusted_n = pd.concat([trusted_n_alter, trusted_n_mono])

    return trusted_p, trusted_n

# Define file paths
rootPath = r".\\"
trainingPath = rootPath+"01 Training sample\\"
validationPath = rootPath+"04 Validation sample\\"
trustedSamplePath = rootPath+"02 Trusted Sample\\"
unlabelPath = rootPath+"03 Current-year unlabel data\\"