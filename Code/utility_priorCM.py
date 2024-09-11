import numpy as np
import pandas as pd

# Get the confusion matrix of trusted samples acquired by alternative rotation in 2020
def getPrioCM_alter(regionName):
    if regionName == 'site_I':
        return np.array([[89875, 10125], [1338, 98662]])

# Get the confusion matrix of trusted samples acquired by monoculture in 2020
def getPrioCM_mono(regionName):
    if regionName == 'site_I':
        return np.array([[9783, 1551], [9740, 90260]])

# Acquire the proportion of monoculture and alternative for negative and positive trusted samples
def get_rotation_Prop(regionName):
    if regionName == 'site_I':
        array = np.array([
            [10667, 2435456],  # Negative trusted sample (Y_T=-1) of monoculture and alternative rotation
            [563756, 2325780]   # Positive trusted sample (Y_T=+1) of monoculture and alternative rotation
        ])
    propArray = np.array([
        [array[0, 0] / (array[0, 0] + array[0, 1]), array[0, 1] / (array[0, 0] + array[0, 1])],
        [array[1, 0] / (array[1, 0] + array[1, 1]), array[1, 1] / (array[1, 0] + array[1, 1])]
    ])

    # propArray
    # Y_T = -1  [[p1, p2], negative proportion between monoculture and alternative
    # Y_T = +1   [p3, p4]] positive proportion between monoculture and alternative
    return propArray

# Safe division function
# In some regions, there is no alternative rotation, so you need the safe division 
# # when deriving the proportion of monoculture and alternative rotation
def sd(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        result = 0
    return result

# Get the combined confusion matrix of trusted samples
def getPrioCM(regionName):
    prop = get_rotation_Prop(regionName)
    CM_mono = getPrioCM_mono(regionName)
    CM_alter = getPrioCM_alter(regionName)

    tN = 100000

    p1 = prop[0, 0]
    p2 = prop[0, 1]

    p1_ = prop[1, 0]
    p2_ = prop[1, 1]

    CM_mono_prop = np.array([
        [sd(CM_mono[0, 0], (CM_mono[0, 0] + CM_mono[0, 1])), sd(CM_mono[0, 1], (CM_mono[0, 0] + CM_mono[0, 1]))],
        [sd(CM_mono[1, 0], (CM_mono[1, 0] + CM_mono[1, 1])), sd(CM_mono[1, 1], (CM_mono[1, 0] + CM_mono[1, 1]))]
    ])
    CM_mono_prop = np.nan_to_num(CM_mono_prop, nan=0)
    
    CM_alter_prop = np.array([
        [sd(CM_alter[0, 0], (CM_alter[0, 0] + CM_alter[0, 1])), sd(CM_alter[0, 1], (CM_alter[0, 0] + CM_alter[0, 1]))],
        [sd(CM_alter[1, 0], (CM_alter[1, 0] + CM_alter[1, 1])), sd(CM_alter[1, 1], (CM_alter[1, 0] + CM_alter[1, 1]))]
    ])
    CM_alter_prop = np.nan_to_num(CM_alter_prop, nan=0)

    CM_mono_new = np.array([
        [tN * p1, tN * p1],
        [tN * p1_, tN * p1_]
    ]) * CM_mono_prop
    
    CM_alter_new = np.array([
        [tN * p2, tN * p2],
        [tN * p2_, tN * p2_]
    ]) * CM_alter_prop

    CM = CM_mono_new + CM_alter_new

    return CM