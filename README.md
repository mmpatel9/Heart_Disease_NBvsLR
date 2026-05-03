# Heart_Disease
A comparison of Naive Bayes and Logistic Regression on predicting whether an individual has heat disease or not.



[View Presentation](https://github.com/mmpatel9/Heart_Disease_NBvsLR/blob/main/ML%20Presentation%20P.pdf)

## Description and motivation

• A heart attack (Cardiovascular diseases) occurs when the flow of blood to the heart muscle suddenly becomes blocked. 17.9 million dying from heart attacks a year. 

•	Solve the binary classification problem of predicting whether someone has heart disease based on certain attributes. 

• These Attributes include Age, Sex, Chest Pain, Trestbps , Chol , FBS, Resting, restecg, Thalach, Exang, Oldpeak, Slope, Ca & Thal 

• Compare and analyse performance of the two models, using performance measure to analyse and evaluate findings.



## Hypothesis Statement:

• Papers we have is based on different data & NB & LR expected to both perform well. 

• If conclusive, i.e., produce similar results able to provide medical advice and attention to those.

•	Papers I have read [1-7] NB-84.25% to 86.6%, LR- 82.56% to 84.85% expect to see the same.



## Training and Evaluation Methodology 

• Date set will be split into train (80%) and test set (20%). 

• A baseline model for each model is created, with training Data. Then be binned and feature reduction applied (Train Set). 

• Choses features are "chol","thalach",'oldpeak','sex','cp','restecg','exang','slope',"binAge",'ca','thal','target’. Age was binned to form ‘binAge’. Highly correlated feat removed

• Respective hyperparameters are applied to the models and will be validated by K-Fold validation (10) 

• 10-Fold validation used to estimate the generalisation error, prevents overfitting. 

• Apply optimization to choose the optimum parameter and validated using 10-Fold validation. 

• Testing the two chosen models with their best parameters on the testing data and contrasting the results to each other.



