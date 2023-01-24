# Predicting Progression of Alzheimer’s Disease (PPAD)

![bi_gru_single](https://user-images.githubusercontent.com/77756538/214337034-759080e6-b94e-4add-9b33-6b19dcace437.svg)

PPAD is a deep learning architecture based on Recurrent Neural Networks (RNN). It consists of a RNN and multi-layer perceptron (MLP). PPAD is designed for early predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) at the next visit for patients. In this architecture, the RNN component learns 𝑥𝑡̂, a latent representation of the longitudinal clinical data up to 𝑡 visits (Eq 1). Then, the MLP model is trained with concatenation of the cross-sectional demographic data (𝐷) and 𝑥𝑡̂ to predict conversion to AD at next visit (Eq 2).

𝑥𝑡̂=𝑅𝑁𝑁(𝑋)                              (1)

𝑦′=𝜎(𝑊1(𝑅𝑒𝐿𝑈(𝑊2(𝑥𝑡̂⊕𝐷)+𝑏2))+𝑏1)        (2)

In Eq 7, 𝑦′ represents the predicted diagnosis, 𝑊1 and 𝑊2 are the trainable linear transformation matrices, and 𝑏1 and 𝑏2 are the bias vectors.

To run PPAD, please have PPAD.ipynb, (dummy_lon_data_train.pkl 

# Predicting Progression of Alzheimer’s Disease Autoencoder (PPAD-AE)

![bi_gru_multiple](https://user-images.githubusercontent.com/77756538/214342401-ac014f9a-9a10-46d5-b7e1-face52854f22.svg)

PPAD-AE is a deep learning architecture based on Recurrent Neural Networks (RNN). It composes of a RNN autoencoder and an MLP. PPAD-AE is designed for early predicting conversion from mild cognitive impairment (MCI) to Alzheimer's disease (AD) at multiple visits ahead for patients. In this architecture, the RNN component learns a latent representation (𝑥𝑡̂) of the longitudinal clinical data up to 𝑡 visits (Eq 1). Then, the latent representation is used by the decoder component to generate representations of multiple visits ahead up to 𝑛 visits. Finally, the MLP model is trained with concatenation of the cross-sectional demographic data (𝐷) and the representation of the last generated visit by the decoder to predict conversion to AD at the (𝑡+𝑛)𝑡ℎ visit (Eq 3).

𝑦′=𝜎(𝑊1(𝑅𝑒𝐿𝑈(𝑊2(𝑥𝑡+(𝑛−1)⊕𝐷)+𝑏2))+𝑏1)  (3)
