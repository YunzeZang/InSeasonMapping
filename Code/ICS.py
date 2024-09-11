import numpy as np
import pandas as pd

class ICS:
    """
     ICS (Integrate classifier transfer and sample transfer for in-season mapping) class.
    
     Attributes:
        prioCM (np.ndarray): Confusion matrix of trusted samples in history.
        sampleNumber (int): Total number of samples to be used after resampling.
        binWidth (float): Width of the bins used for discretizing the prediction probabilities.
        balancedNumber (int): Number of samples to balance between positive and negative classes.
        hist (np.ndarray): Bins edges.
        RSP_t_p (np.ndarray): Resampling sample proportion for positive trusted samples.
        RSP_t_n (np.ndarray): Resampling sample proportion for negative trusted samples.
        RSP_c (np.ndarray): Resampling sample proportion for classified samples.
    """
    def __init__(self, prioCM, sampleNumber=6000, binWidth=0.05, balancedNumber = 25000):
        self.sampleNumber = sampleNumber
        self.binWidth = binWidth
        self.balancedNumber = balancedNumber
        self.hist = np.arange(0, 1+binWidth, binWidth)
        
        self.prioCM = prioCM

        self.RSP_t_p = None
        self.RSP_t_n = None 
        self.RSP_c = None

    def estimate(self, y_predicted_p_test,y_predicted_n_test,y_predicted_c,binWidth = 0.05):
        """
        Estimate the weights and resampling proportions for trusted and classified samples.
        
        Args:
            y_predicted_p_test (np.ndarray): Predicted probabilities for positive test samples.
            y_predicted_n_test (np.ndarray): Predicted probabilities for negative test samples.
            y_predicted_c (np.ndarray): Predicted probabilities for classified samples.
            binWidth (float, optional): Width of the bins used for discretizing the prediction probabilities. Defaults to 0.05.
        """
        # balance the positive and negative trusted samples and classified samples
        y_predicted_p_classified = self.overSampling(pd.DataFrame(y_predicted_c[y_predicted_c>=0.5]),self.balancedNumber)
        y_predicted_n_classified = self.overSampling(pd.DataFrame(y_predicted_c[y_predicted_c<0.5]),self.balancedNumber)
        y_predicted_classified = pd.concat([y_predicted_p_classified,y_predicted_n_classified]).values

        y_predicted_p_test = self.overSampling(pd.DataFrame(y_predicted_p_test),self.balancedNumber).values
        y_predicted_n_test = self.overSampling(pd.DataFrame(y_predicted_n_test),self.balancedNumber).values

    # calculate weight and RSP of samples in each bin using "test" part of trusted samples

        # statistical distribution of trusted samples and classified samples
        probDis_c = []
        probDis_t_p = []
        probDis_t_n = []

        for k,_ in enumerate(self.hist[:-1]):
            beg_of_bin = self.hist[k]
            end_of_bin = self.hist[k+1]
            if end_of_bin == 1:
                end_of_bin = 1+1e-5
            probDis_c.append(y_predicted_classified[(y_predicted_classified>=beg_of_bin)&(y_predicted_classified<end_of_bin)].shape[0])
            probDis_t_p.append(y_predicted_p_test[(y_predicted_p_test>=beg_of_bin)&(y_predicted_p_test<end_of_bin)].shape[0])
            probDis_t_n.append(y_predicted_n_test[(y_predicted_n_test>=beg_of_bin)&(y_predicted_n_test<end_of_bin)].shape[0])

        # normalize the distribution
        probDis_c_norm = np.array(probDis_c)/np.sum(probDis_c)
        probDis_t_p_norm = np.array(probDis_t_p)/(np.sum(probDis_t_p)+np.sum(probDis_t_n))
        probDis_t_n_norm = np.array(probDis_t_n)/(np.sum(probDis_t_p)+np.sum(probDis_t_n))

        #calculate the P(Y|g(X)∈B_i) and P(Y_T|g(X)∈B_i)
        probList_est_p,probList_t_p = self.deriveP(y_predicted_p_test,y_predicted_n_test,Y=1)
        probList_est_n,probList_t_n = self.deriveP(y_predicted_p_test,y_predicted_n_test,Y=0)

        # calculate weight of trusted samples and classifed samples in each bin
        weight_t_p = (probList_est_p+1e-5)/(probList_t_p+1e-5) # P(Y=+1|g(X)∈B_i)/P(Y_T=+1|g(X)∈B_i)
        weight_c_p = probList_est_p/1 # P(Y=+1|g(X)∈B_i)/P(Y_C=+1|g(X)∈B_i), where the P(Y_C=+1|g(X)∈B_i)=1

        weight_t_n = (probList_est_n+1e-5)/(probList_t_n+1e-5) # P(Y=-1|g(X)∈B_i)/P(Y_T=-1|g(X)∈B_i)
        weight_c_n = probList_est_n/1 # P(Y=-1|g(X)∈B_i)/P(Y_C=-1|g(X)∈B_i), where the P(Y_C=-1|g(X)∈B_i)=1

        # remove anomaly weight
        weight_t_p[weight_t_p<0]=0
        weight_t_n[weight_t_n<0]=0
        weight_c_n[weight_c_n<0]=0
        weight_c_p[weight_c_p<0]=0
    
    
        # calculate RSP (resampling sample proportion)
        self.RSP_t_p = np.array(weight_t_p*probDis_t_p_norm/((weight_t_p*probDis_t_p_norm).sum()+(weight_t_n*probDis_t_n_norm).sum()),dtype=float)
        self.RSP_t_n = np.array(weight_t_n*probDis_t_n_norm/((weight_t_p*probDis_t_p_norm).sum()+(weight_t_n*probDis_t_n_norm).sum()),dtype=float)
        
        RSP_c_p = np.array(weight_c_p*probDis_c_norm/((weight_c_p*probDis_c_norm).sum()+(weight_c_n*probDis_c_norm).sum()),dtype=float)
        RSP_c_n = np.array(weight_c_n*probDis_c_norm/((weight_c_p*probDis_c_norm).sum()+(weight_c_n*probDis_c_norm).sum()),dtype=float)
        self.RSP_c = np.concatenate((RSP_c_n[:10],RSP_c_p[10:]),dtype=float)

        # replace positive classified samples using trusted samples when classification probability<50% 
        RSP_c_p[10:]=0
        self.RSP_t_p = self.RSP_t_p+RSP_c_p

        # replace negative classified samples using trusted samples when classification probability<50% 
        RSP_c_n[:10]=0
        self.RSP_t_n = self.RSP_t_n+RSP_c_n
        
    def weight(self,trusted_sample_p,trusted_sample_n,classified_sample):
        """
        Apply the estimated weights to resample the trusted and classified samples.
        
        Args:
            trusted_sample_p (pd.DataFrame): DataFrame containing positive trusted samples.
            trusted_sample_n (pd.DataFrame): DataFrame containing negative trusted samples.
            classified_sample (pd.DataFrame): DataFrame containing classified samples.
        
        Returns:
            tuple: A tuple containing the resampled DataFrame and labels.
        """
        sampledArray_t = pd.DataFrame({})
        sampledLabel_t = np.array([])

        sampledArray_u = pd.DataFrame({})
        sampledLabel_u = np.array([])

        N_t_p = np.array(self.RSP_t_p*self.sampleNumber/2,dtype= int)
        N_t_n = np.array(self.RSP_t_n*self.sampleNumber/2,dtype = int)
        N_c = np.array(self.RSP_c*self.sampleNumber/2, dtype= int)

        for k,_ in enumerate(self.hist[:-1]):
            beg_of_bin = self.hist[k]
            end_of_bin = self.hist[k+1]

            currentSampleArray_t_p = trusted_sample_p[(trusted_sample_p['prob']>=beg_of_bin)*(trusted_sample_p['prob']<end_of_bin)]
            currentSampleArray_t_n = trusted_sample_n[(trusted_sample_n['prob']>=beg_of_bin)*(trusted_sample_n['prob']<end_of_bin)]
            currentSampleArray_his = classified_sample[(classified_sample['prob']>=beg_of_bin)*(classified_sample['prob']<end_of_bin)]

            if currentSampleArray_t_p.shape[0]>0:
                newSample = self.overSampling(currentSampleArray_t_p,N_t_p[k])
                sampledArray_t = pd.concat([sampledArray_t,newSample])
                sampledLabel_t = np.concatenate((sampledLabel_t,np.ones(newSample.shape[0])))

            if currentSampleArray_t_n.shape[0]>0:
                newSample = self.overSampling(currentSampleArray_t_n,N_t_n[k])
                sampledArray_t = pd.concat([sampledArray_t,newSample])
                sampledLabel_t = np.concatenate((sampledLabel_t,np.zeros(newSample.shape[0])))

            if currentSampleArray_his.shape[0]>0:
                newSample = self.overSampling(currentSampleArray_his,N_c[k])
                sampledArray_u = pd.concat([sampledArray_u,newSample])
                sampledLabel_u = np.concatenate((sampledLabel_u,np.zeros(newSample.shape[0]) if k < self.hist.shape[0]/2 else np.ones(newSample.shape[0])))

        # merge the weighted samples
        X_train_w = pd.concat([sampledArray_t,sampledArray_u])
        y_train_w = np.concatenate((sampledLabel_t,sampledLabel_u))

        return X_train_w, y_train_w
    
    def deriveP(self,N_Yt_p,N_Yt_n,Y):
        """
        Calculate the conditional probabilities P(Y|g(X)∈B_i) and P(Y|g(X)∈B_i) given the bin.
        
        Args:
            N_Yt_p (np.ndarray): Number of positive trusted samples in each bin.
            N_Yt_n (np.ndarray): Number of negative trusted samples in each bin.
            Y (int): The target class label (0 or 1).
        
        Returns:
            tuple: A tuple containing the calculated probabilities.
        """
        prioCM = self.prioCM
        hist = self.hist
        def deriveP_Y_gX(p_Yt1_given_Y0,p_Yt0_given_Y1,p_Yt_given_gX,Yt): #calculate P(Y|g(X)∈B_i) in a specific bin
        
            if Yt == 1:
                P_Y_gX = (p_Yt_given_gX-p_Yt1_given_Y0+1e-5)/(1-p_Yt1_given_Y0-p_Yt0_given_Y1+1e-5)
            if Yt == 0:
                P_Y_gX = (1-p_Yt_given_gX-p_Yt0_given_Y1+1e-5)/(1-p_Yt0_given_Y1-p_Yt1_given_Y0+1e-5)
            
            return P_Y_gX

        P_Y_given_gX = np.zeros(hist.shape[0]-1)
        P_Yt_given_gX = np.zeros(hist.shape[0]-1)

        p_Yt0_given_Y1 = prioCM[0,1]/(prioCM[0,1]+prioCM[1,1])
        p_Yt1_given_Y0 = prioCM[1,0]/(prioCM[1,0]+prioCM[0,0])

        for i in range(0,hist.shape[0]-1):
            sP = hist[i]
            eP = hist[i+1]
            if eP == 1:
                eP = 1+1e-5

            N_Yt_p_Bi  = N_Yt_p[(N_Yt_p>=sP)&(N_Yt_p<eP)]
            N_Yt_n_Bi  = N_Yt_n[(N_Yt_n>=sP)&(N_Yt_n<eP)]

            P_Yt_1_given_gX = N_Yt_p_Bi.shape[0]/(N_Yt_p_Bi.shape[0]+N_Yt_n_Bi.shape[0]) #p(Yt=1|g(X)∈B_i)
            
            P_Yt_given_gX[i] = P_Yt_1_given_gX if Y==1 else 1-P_Yt_1_given_gX

            P_Y_given_gX[i] = deriveP_Y_gX(p_Yt1_given_Y0,p_Yt0_given_Y1,P_Yt_1_given_gX,Y) #p(Y|g(X)∈B_i)

        return P_Y_given_gX, P_Yt_given_gX

    def overSampling(self,sample,number):
        """
        Perform oversampling of the provided sample to reach the specified number of samples.
        
        Args:
            sample (pd.DataFrame): DataFrame containing the samples to be oversampled.
            number (int): Desired number of samples.
        
        Returns:
            pd.DataFrame: Oversampled DataFrame.
        """
        number = int(number)
        if number<=0:
            print('sample Number<0!')
            return sample.sample(0)
        if number<=sample.shape[0]:
            return sample.sample(number)
        if number>sample.shape[0]:
            if sample.shape[0] == 0:
                print('sample size==0')
                return sample
            else:
                print('replace sampling')
                return sample.sample(n=number,replace=True)

