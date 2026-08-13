
</think>

# Soruce code and data for "Integrating classifier transfer and sample transfer strategies for in-season crop mapping based on sample weighting techniques"

This repository contains the source code and example data from Site I. The provided code allows for the repetition and evaluation of four classification methods: HSC, TSC, CSC, EWSC and the proposed OWSC.

# Available Scripts

Main scripts:

● Model_Baseline.py: This script trains the HSC, TSC, or CSC.

● Model_OWSC.py: This script trains the OWSC or EWSC.

● Evaluation.py: This script evaluates trained classifiers (supports different classification models and different methods).

Utility scripts:

● utility_prepareData.py: Prepare data

● utility_prioCM.py: Load confusion matrix in the history and calculate the error rates of trusted samples. 

● utility_trainInSeasonRF.py: Functions for trainning RF and SVM during mid-season.

● utility_trainDL.py: Functions for trainning deep learning models during mid-season.

● ICS Class.py: A Plug-and-Play Solution for the proposed method

● prioCM.xlsx: file for saving confusion matrix, proportion of different type of rotation.

The ICS class implements the weighting process as a plug-in for any classifier. For details on how to use it, please refer to the "02_weighted script". This class supports both machine learning and deep learning classifiers with softmax output probabilities.

# Google Earth Engine app for results visualization and example data

We provide a GEE app for comparing mapping results of different methods across different sites and years (https://ee-zaggyunze.projects.earthengine.app/view/owsc)

The example training data is available at https://drive.google.com/file/d/1X3mj7xjg7j5suDdKehDBlXSAprl1rtrW/view?usp=sharing

The mapping results of OWSC in Sites I to VI is available at https://drive.google.com/file/d/1h-_AKzMXXZdTx7ih9Oknrn__TOdwhd6r/view?usp=sharing

# How to use?
1. Modify the file path in utility_prepareData.py. Make sure orgnize data using format in utility_prepareData.py
2. Set parameters and run Model_Baseline.py to train a HSC
3. Run Model_OWSC.py to train OWSC
4. Evaluate models using Evaluation.py

# Environment Setup
We test these scripts in following version:
●  python 3.8.18
●  scikit-learn 1.2.2

# Contact
If you have any questions, feel free to contact me at: zangyunze@mail.bnu.edu.cn.
