# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a bank term deposit.
The best performing model was a VotingEnsemble AutoML model.

## Scikit-learn Pipeline
- Training of a Logistic Regression model is implemented in train.py including dataset cleaning
- Data is accessed through Dataset created by TabularDatasetFactory from delimited csv files in train.py
- Random Parameter Sampling is used for HyperDrive based hyperparameter tuning of C (inverse regularization strength) and max iterations enabling a quick search in the hyperparameter space  
- Early stopping Bandit Policiy ensures that any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated

## AutoML
VotingEnsemble AutoML model implements soft-voting, which uses weighted averages of multiple fitted models from the previous child runs. The Voting Ensemble model has achieved accuracy of 0.916. The best non-ensemble AutoML model was MaxAbsScaler, LightGBM which is a boosted tree model with accuracy 0.915.

## Pipeline comparison
The relatively simple Scikit-learn binary logistic model has achieved good performance despite the linear relationship between the predictor variables and the log odds of the event that y=1. The more advanced best VotingEnsemble AutoML model has achieved slightly better accuracy of 0.916 vs 0.910 of Logistic Regression. 

## Future work
Potential future improvements include but are not limited to:
- Addressing class imbalance in training data set
  - AutoML Class balancing detection data guardrail has alerted since the size of the smallest class in the training data set is 3692 out of 32950.
  - Imbalanced data can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class.
  - Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData
- Hyperparameter tuning of the best model
  - By optimizing the hyperparameters of the best AutoML model further improvements are expected in model performance
- Exploring feature importance
  - This would be helpful to get futher insights into this data set
- Analyze confusion matrices
  - To get a better understanding of how and why some models might fail to predict the true classes
- Training models longer or on larger cluster
  - For best results larger models and fine grained hyperparameter optimization might need more computing cycles


```

```
