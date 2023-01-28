# Predicting Progression of Alzheimer’s Disease (PPAD)

![bi_gru_single](https://user-images.githubusercontent.com/77756538/214337034-759080e6-b94e-4add-9b33-6b19dcace437.svg)

PPAD is a deep learning architecture based on Recurrent Neural Networks (RNN). It consists of a RNN and multi-layer perceptron (MLP). PPAD is designed for early predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) at the next visit for patients.
  
In this architecture, the RNN component learns 𝑥𝑡̂, a latent representation of the longitudinal clinical data up to 𝑡 visits. Then, the MLP model is trained with concatenation of the cross-sectional demographic data (𝐷) and 𝑥𝑡̂ to predict conversion to AD at next visit.

# Predicting Progression of Alzheimer’s Disease Autoencoder (PPAD-AE)

![bi_gru_multiple](https://user-images.githubusercontent.com/77756538/214342401-ac014f9a-9a10-46d5-b7e1-face52854f22.svg)

PPAD-AE is a deep learning architecture based on Recurrent Neural Networks (RNN). It composes of a RNN autoencoder and an MLP. PPAD-AE is designed for early predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) at multiple visits ahead for patients. 

In this architecture, the RNN component learns a latent representation (𝑥𝑡̂) of the longitudinal clinical data up to 𝑡 visits. Then, the latent representation is used by the decoder component to generate representations of multiple visits ahead up to 𝑛 visits. Finally, the MLP model is trained with concatenation of the cross-sectional demographic data (𝐷) and the representation of the last generated visit by the decoder to predict conversion to AD at the (𝑡+𝑛)𝑡ℎ visit.

# Parameter learning and evaluation metrics

To increase the prediction’s sensitivity for both architectures, all trainable parameters for the RNN, RNN autoencoder, and MLP were learned in an integral way using a customized binary cross-entropy loss function to give more weight on predicting conversion to AD cases. We seek by using this customized loss function to minimize the false negative cases while predicting diagnosis of the future visit which leads to increased sensitivity of the predictive model.
  
RNN cell, number of epochs, batch size, dropout rate, and L2 regularization are the hyperparameters that have been tuned. For model evaluation, F2 score and sensitivity were used.

# Datasets and input format

We evaluated the proposed architectures in two experimental setups. In the first setup, Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset was utilized to train and test the proposed architectures using the longitudinal multi-modal and the cross-sectional demographic data. In the second setup, the models were trained on the entire ADNI longitudinal and cross-sec-tional data and tested on National Alzheimer’s Coordinating Center (NACC) dataset.
  
To get ADNI, you need to request an access for it https://adni.loni.usc.edu/
  
To get NACC, you need to request an access for it https://naccdata.org/
  
The training and test longitudinal data format is given where data is stored as a list containing 3 dimensionals tensors such as [number of samples , number of visits , number of longitudinal feature in each vist].
  
The training and test demographic data format is given where data is stored as a list containing other lists such that each inner list represents demographic features for one sample.
    
The training and test label data format is given where data is stored as a list containing 3 dimensionals tensors such as [number of samples , number of visits , 1] where the third dimension can be 0 (MCI) or 1 (Dementia).

Both PPAD sample dataset and PPAD-AE folders contain the following pkl files:

 - longitudinal_data_train.pkl which represents longitudinal training data

 - label_train.pkl which represents labels of traing data 

demographic_data_train.pkl which represents demographic training data

longitudinal_data_test.pkl which represents longitudinal test data

label_test.pkl which represents labels of test data

demographic_data_test.pkl which represents demographic test data
  
 # Compitability
 
 All codes are compatible with tensorflow version 2.4.1, keras version 2.4.3 and Pyhton 3.8.5.
 
 # How to run PPAD
