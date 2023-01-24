# Predicting Progression of Alzheimer’s Disease (PPAD)

![bi_gru_single](https://user-images.githubusercontent.com/77756538/214337034-759080e6-b94e-4add-9b33-6b19dcace437.svg)

  PPAD is a deep learning architecture based on Recurrent Neural Networks (RNN). It consists of a RNN and multi-layer perceptron (MLP). PPAD is designed for early predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) at the next visit for patients.
  
  In this architecture, the RNN component learns 𝑥𝑡̂, a latent representation of the longitudinal clinical data up to 𝑡 visits (Eq 1). Then, the MLP model is trained with concatenation of the cross-sectional demographic data (𝐷) and 𝑥𝑡̂ to predict conversion to AD at next visit (Eq 2).

𝑥𝑡̂=𝑅𝑁𝑁(𝑋) ...................................................(1)

𝑦′=𝜎(𝑊1(𝑅𝑒𝐿𝑈(𝑊2(𝑥𝑡̂⊕𝐷)+𝑏2))+𝑏1) .............................(2)

In Eq 7, 𝑦′ represents the predicted diagnosis, 𝑊1 and 𝑊2 are the trainable linear transformation matrices, and 𝑏1 and 𝑏2 are the bias vectors.

To run PPAD, please have PPAD.ipynb, (dummy_lon_data_train.pkl 

# Predicting Progression of Alzheimer’s Disease Autoencoder (PPAD-AE)

![bi_gru_multiple](https://user-images.githubusercontent.com/77756538/214342401-ac014f9a-9a10-46d5-b7e1-face52854f22.svg)

PPAD-AE is a deep learning architecture based on Recurrent Neural Networks (RNN). It composes of a RNN autoencoder and an MLP. PPAD-AE is designed for early predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) at multiple visits ahead for patients. 

  In this architecture, the RNN component learns a latent representation (𝑥𝑡̂) of the longitudinal clinical data up to 𝑡 visits (Eq 1). Then, the latent representation is used by the decoder component to generate representations of multiple visits ahead up to 𝑛 visits. Finally, the MLP model is trained with concatenation of the cross-sectional demographic data (𝐷) and the representation of the last generated visit by the decoder to predict conversion to AD at the (𝑡+𝑛)𝑡ℎ visit (Eq 3).

𝑦′=𝜎(𝑊1(𝑅𝑒𝐿𝑈(𝑊2(𝑥𝑡+(𝑛−1)⊕𝐷)+𝑏2))+𝑏1) ........................(3)

# Parameter learning and evaluation metrics

  To increase the prediction’s sensitivity for both architectures, all trainable parameters for the RNN, RNN autoencoder, and MLP were learned in an integral way using a customized binary cross-entropy loss function (Eq 4) to give more weight on predicting conversion to AD cases. We seek by using this customized loss function to minimize the false negative cases while predicting diagnosis of the future visit which leads to increased sensitivity of the predictive model.
  
𝐿𝑜𝑠𝑠=−1/𝑁Σ(𝛼∙(𝑦∙𝑙𝑜𝑔𝑦′))+((1−𝛼)∙(1−𝑦)∙𝑙𝑜𝑔(1−𝑦′)) ................(4)

In Eq 4, 𝛼 is a real number between 0 and 1 to define the relative weight of positive prediction, 𝑦 is the true diagnosis, and 𝑦′ is the predicted diagnosis.

  In this study, we set 𝛼 to 0.7. Based on the proposed customized loss function, all trainable parameters for the RNN, RNN autoencoder, and MLP (Eq 2 and 3) were updated while training models in the backpropagation. For optimization, all models were trained using Adaptive Moment Estimation (Adam) optimizer and the learning rate was set to 0.001.
  
  RNN cell, number of epochs, batch size, dropout rate, and L2 regularization are the hyperparameters that have been tuned. For model evaluation, F2 score (Eq 5) and sensitivity were used.
  
𝐹𝛽 = (1+𝛽2) ∙ (precision∙recall) / (𝛽2.precision+recall) .......(5)

In Eq 5, recall is considered 𝛽 times more important than precision. In this study, 𝛽 was set to 2.

# How to 
