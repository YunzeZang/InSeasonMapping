# Soruce code and example data for "Integrating classifier transfer and sample transfer strategies for in-season crop mapping based on instance weighting techniques"

This repository contains the source code and example data from Site I. The provided code allows for the repetition and evaluation of four classification methods: HSC, TSC, CSC, and the proposed method.

# Available Scripts
● 01_baselineModel: This script trains and evaluates the HSC, TSC, or CSC classifiers.

● 02_weighted: This script trains the weighted classifier as proposed in the accompanying research paper.

● utility_prepareData: Prepare data

● utility_prioCM: Restore confusion matrix of trusted samples in the history. 

● utility_trainInSeasonRF: Functions for trainning RF during mid-season.

● ICS Class: A Plug-and-Play Solution for Weighting

We provide an ICS class in the file ICS.py, which implements the weighting process as a plug-in for any classifier. For details on how to use it, please refer to the "02_weighted script". To apply this to a new 	region, simply prepare the data as described in the script and modify your classifier. This class supports both machine learning and deep learning classifiers with softmax output probabilities.

# Example data

The example data is available at https://drive.google.com/file/d/1X3mj7xjg7j5suDdKehDBlXSAprl1rtrW/view?usp=sharing

# Environment Setup
We test these scripts in following version:
●  python 3.8.18
●  scikit-learn 1.2.2

# Contact
If you have any questions, feel free to contact me at: zangyunze@mail.bnu.edu.cn.
