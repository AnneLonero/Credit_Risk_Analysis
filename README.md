# Credit_Risk_Analysis

## Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumbered risky loans. Therefore, we will need to employ different techniques to train an evaluate models with unbalanced classes. In this analysis, we will be using the credit card dataset from LendingClub, a peer-to-peer lending services company to do the following:

* Oversample the data using the RandomOverSample and SMOTE algorithms
* Undersample the data using the ClsuterCentroids algorithm
* Use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm
* Compare two machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier
    to predict credit risk
    
At the end, based on the evaluation of these models performance, we will make a recommendation on whether they could be used to predict credit risk.    

## Results

After import and examine our dataset, we found that our targeted data is highly skewed toward low risk, 68470 datapoints versus 347 datapoints. It become more prevalent when we split the targeted data to train and test set as shown below.

![image](https://user-images.githubusercontent.com/114631804/228913423-07c6f4cb-a51e-4723-93b6-c985ce5921f4.png)

![image](https://user-images.githubusercontent.com/114631804/228914817-f3abc41b-5192-4e52-bf02-ea084943af6d.png)

As a result, we attempted to resample the dataset to produce better models using different techniques of Oversampling and Undersampling.

### Naive RandomOverSampler model
This techinuqes involves randomly duplicating examples from the minority class and adding them to the training dataset, which would produce "more balanced" dataset. After applied the technique, we came up with 51352 datapoints for low risk and 51352 datapoints for high risk.

![image](https://user-images.githubusercontent.com/114631804/228916591-f7a29af8-bf47-458d-80f0-8e28847a0c9f.png)

However, we found that the balanced accuracy score is only about 63% for this model.

![image](https://user-images.githubusercontent.com/114631804/228739596-66cc1e59-5b69-45fb-ad73-70e6ba0b4bc5.png)

![image](https://user-images.githubusercontent.com/114631804/228739917-2bde3069-be6a-4a80-8c2b-3d0fe0b7ade4.png)

The high risk precision is 1% while the sensitivity is 57% which makes F1 score is 0.02
![image](https://user-images.githubusercontent.com/114631804/228739978-f9999563-35d6-4d85-a4a8-86a137e66c64.png)


### SMOTE (Synthetic Minority OverSampling Technique) model
SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line. This technique also produces 513252 datapoints each for low and high risk.

The balanced accuracy is also only around 63%

![image](https://user-images.githubusercontent.com/114631804/228740255-fe08c879-df07-4dee-9bb4-6520d89de12a.png)

![image](https://user-images.githubusercontent.com/114631804/228740364-afb6e33c-2a2c-4859-9fe0-00ce1b821d70.png)

The precision for high risk is 1% and sensitivity is 62% while F1-score is 0.02.
![image](https://user-images.githubusercontent.com/114631804/228740411-911a5b9c-5fde-4c59-b9ad-fcbfde98293f.png)

### ClusterCentroids (Undersampling) model

According to https://imbalanced-learn.org/, "Method that under samples the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm. This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster to the majority class abd using the coordinates of the N cluster centroids as the new majority samples".

Using this method, we got 260 datapoints each for high risk and low risk.
![image](https://user-images.githubusercontent.com/114631804/228928143-545c5967-987c-44bb-8edf-7d607a87a80f.png)

We found the balanced accuracy score is decreased to 51%

![image](https://user-images.githubusercontent.com/114631804/228740492-bb4ce38c-a37b-4d7b-a9ce-8a12c4e85dd7.png)

![image](https://user-images.githubusercontent.com/114631804/228740541-3e2f0ad0-9f9e-4eae-987c-9b9059e7c1c1.png)

The precision for high risk is pretty low as well at 1% with sensitivity at 56%, and F1-score is only 0.01.
![image](https://user-images.githubusercontent.com/114631804/228740607-d0deb173-b2ed-4215-90a0-f4a62bf3ad38.png)

### SMOTEENN (Combination Over and Under Sampling) model
This method used SMOTE and Edited Nearest Neighbours. According to https://imbalanced-learn.org/, "SMOTE can generate noisy samples by interpolating new points between marginal outliers and inliers. This issue can be solved by cleaning the space resulting from over-sampling".

After resampled the dataset using SMOTEENN, WE GOT 51351 datapoints for high risk and 46389 datapoints for low risk.

![image](https://user-images.githubusercontent.com/114631804/228932072-c7074bca-bd6e-493c-85fb-8a79b949d356.png)

The balanced accuracy score is 62% for this model

![image](https://user-images.githubusercontent.com/114631804/228907762-057b4c66-c59d-48a0-8918-3ee1f9c61819.png)

The precision is still at 1% but the sensitivity is slighty improved to 70% and F1-score is 0.02.
![image](https://user-images.githubusercontent.com/114631804/228907945-258902fc-e11d-44a3-ac28-6a0ed089b836.png)

### BalancedRandomForestClassifier model
This is an ensemble method in which each tree of the forest will be provided a balanced bootstrap sample.

The balanced accuraccy score improved to 79%.

![image](https://user-images.githubusercontent.com/114631804/228937356-2bcb8022-ef42-446c-bd68-a810032d4458.png)

The precision for high risk is at 4%, sensitivity is 67% and F1-score is 0.07.
![image](https://user-images.githubusercontent.com/114631804/228937705-9ee3ff2e-246b-4f89-be90-196b5056f77f.png)

### EasyEnsembleClassifier model
This method allows to bag AdaBoost learners which are trained on balanced bootstrap samples. An Adaboost on the other hand, according to https://scikit-learn.org/, "is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases".

The balanced accuracy score for this model is 92.5%

![image](https://user-images.githubusercontent.com/114631804/228940816-36cb435a-1a66-4d49-bd3c-185eed0f1977.png)

The precision is way improved at 7% with the sensitivity up to 91%, F1-score is 0.14.
This proves to be the best model we have evaluated so far.

![image](https://user-images.githubusercontent.com/114631804/228941255-e996383b-d46f-4de1-a904-32b3813a35c1.png)

## Summary

In conclusion, none of the model is perfect since all the models show low precision in determining  if a credit risk is high. The Ensemble models shows a lot of improvement, especially on the sensitivity of the high risk credits. The EasyEnsembleClassifier model shows the most potential with a recall of 92.5%. However, with a low precision, a lot of low risk are still falsely detected as high risk and vice versa, which would impact the bank potential revenue. Therefore, I would recommend further collect data to improve the balance of our dataset and to better train our model. This way we will have a better model in the future to precisely predict the credit risks for our creditors.
