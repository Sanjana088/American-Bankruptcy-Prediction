# American-Bankruptcy-Prediction

## Overview
This task focuses on building machine learning models to predict whether a company will go bankrupt based on its financial features.The goal was to handle a highly imbalanced dataset, explore classification models, evaluate their performance, and reflect on their strengths and weaknesses.

## Dataset
Total companies: 8,262
Features: 23 financial indicators
Target: Bankrupt (1) or Alive (0)
Class distribution: ~6.55% bankrupt, ~93.45% alive
Missing data: 13 features contained missing values, handled by median imputation
Split: 80% training, 20% testing, using stratified sampling to maintain class distribution


## Preprocessing
I prepared the data with the following steps:
Handled missing values using median imputation.
Standardised numerical features to improve model performance.
Split the dataset into training and testing sets (80:20) with stratified sampling.


## Models Used
I implemented and compared two supervised learning models:

1. Random Forest Classifier
200 decision trees
Maximum depth: 20
Random state fixed for reproducibility

2. Multilayer Perceptron (MLP)
Three hidden layers: 128, 64, 32 neurons
ReLU activation
Adam optimizer
Early stopping to prevent overfitting

## Results
Metric	Random Forest	MLP
Accuracy	93.99%	93.53%
Precision (Bankrupt)	90.63%	83.33%
Recall (Bankrupt)	9.21%	5.08%
ROC-AUC	0.921	0.768


## Observations:
Both models achieved similar accuracy, but accuracy alone was misleading due to the class imbalance.
Random Forest outperformed MLP in precision and recall for the bankrupt class.
Recall remained very low for both models, meaning many bankrupt companies were not detected.
ROC-AUC showed that Random Forest had much stronger discriminative power.

## Key Insights
High accuracy does not imply good performance when the data is imbalanced.
Precision for the minority class (bankrupt companies) was strong, but low recall indicated the model struggled to catch all true bankrupt cases.
Random Forest was more robust overall, likely due to its ensemble nature and ability to capture non-linear relationships.


## Future Improvements
To improve recall and overall performance, I suggested several approaches:
Resampling techniques like SMOTE to balance the dataset.
Class weighting to penalise misclassification of the minority class.
Focal loss to focus learning on hard-to-classify bankrupt cases.
Threshold tuning to trade off precision and recall based on business needs.
