# Credit_Risk_Analysis

## Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumbered risky loans. Therefore, we will need to employ different techniques to train an evaluate models with unbalanced classes. In this analysis, we will be using the credit card credit dataset from LendingClub, a peer-to-peer lending services company to do the following:

    * Oversample the data using the RandomOverSample and SMOTE algorithms
    * Undersample the data using the ClsuterCentroids algorithm
    * Use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm
    * Compare two machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier
    to predict credit risk
    
At the end, based in the evaluation of these models performance, we will make a recommendation on whether they could be used to predict redit risk.    

## Results

### Naive RandomOverSampler model

![image](https://user-images.githubusercontent.com/114631804/228739596-66cc1e59-5b69-45fb-ad73-70e6ba0b4bc5.png)

![image](https://user-images.githubusercontent.com/114631804/228739917-2bde3069-be6a-4a80-8c2b-3d0fe0b7ade4.png)

![image](https://user-images.githubusercontent.com/114631804/228739978-f9999563-35d6-4d85-a4a8-86a137e66c64.png)


### SMOTE (Synthetic Minority OverSampling Technique) model

![image](https://user-images.githubusercontent.com/114631804/228740255-fe08c879-df07-4dee-9bb4-6520d89de12a.png)

![image](https://user-images.githubusercontent.com/114631804/228740364-afb6e33c-2a2c-4859-9fe0-00ce1b821d70.png)

![image](https://user-images.githubusercontent.com/114631804/228740411-911a5b9c-5fde-4c59-b9ad-fcbfde98293f.png)

### ClusterCentroids (Undersampling) model

![image](https://user-images.githubusercontent.com/114631804/228740492-bb4ce38c-a37b-4d7b-a9ce-8a12c4e85dd7.png)

![image](https://user-images.githubusercontent.com/114631804/228740541-3e2f0ad0-9f9e-4eae-987c-9b9059e7c1c1.png)

![image](https://user-images.githubusercontent.com/114631804/228740607-d0deb173-b2ed-4215-90a0-f4a62bf3ad38.png)

### SMOTEENN (Combination Over and Under Sampling) model


### BalancedRandomForestClassifier model

### EasyEnsembleClassifier model

## Summary
