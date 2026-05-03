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

Analysis and Critical Evaluation

• Both models have high accuracy in the training phase. The Base model was
improved by applying regularization techniques , with LR model having higher
accuracy in the training phase.

• Applying both models to the testing data we see that there was a decrease in
accuracy, with LR decreasing quite significantly from 92.07 to 80% I and NB
dropped from 91.71% to 90% in terms of accuracy. This tells us that we
overfitted the training data, especially for LR and more significantly, NB
outperformed LR, with higher performance, which most papers agreed with, but
not the magnitude.

• This tells us that Looking into this further; we can see that ‘Recall’ was
significantly greater for NB than LR, shows that it to detect people with heart
disease efficiently, highlighting the fact that logistic regression has been over
trained.

• ROC curve, figure4, shows that NB has a higher AUC than, LR. Shows that it fits
dataset better than LR.

• Comparing this to paper[3]; we see that there are indifference in results,[NB-
90.78%, LR-92.76%], especially for LR which could be since the data used where
different as well as that the way the data in terms of features were collected was
also be different.

• Results for NB, where like those results [3], even though the data is different,
shows, NB performs well with different datasets.





