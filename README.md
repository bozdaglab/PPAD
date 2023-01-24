# Predicting Progression of Alzheimer’s Disease (PPAD)

![bi_gru_single](https://user-images.githubusercontent.com/77756538/214337034-759080e6-b94e-4add-9b33-6b19dcace437.svg)

PPAD is a deep learning architecture based on Recurrent Neural Networks (RNN). PPAD is designed for early predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) at the next visit. In this architecture, the RNN component learns 𝑥𝑡̂, a latent representation of the longitudinal clinical data up to 𝑡 visits (Eq 6). Then, the MLP model is trained with concatenation of the cross-sectional demographic data (𝐷) and 𝑥𝑡̂ to predict conversion to AD at next visit (Eq 7).

𝑥𝑡̂=𝑅𝑁𝑁(𝑋)  (6)

𝑦′=𝜎(𝑊1(𝑅𝑒𝐿𝑈(𝑊2(𝑥𝑡̂⊕𝐷)+𝑏2))+𝑏1)  (7)

In Eq 7, 𝑦′ represents the predicted diagnosis, 𝑊1 and 𝑊2 are the trainable linear transformation matrices, and 𝑏1 and 𝑏2 are the bias vectors.

To run PPAD, please have PPAD.ipynb, (dummy_lon_data_train.pkl 
