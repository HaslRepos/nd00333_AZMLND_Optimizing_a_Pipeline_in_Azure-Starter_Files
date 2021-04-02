# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

A diagram illustrating the steps of this project is shown below:

![alt text](https://github.com/HaslRepos/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/creating-and-optimizing-an-ml-pipeline.png "Optimizing an ML Pipeline")

Source: Machine Learning Engineer with Microsoft Azure Nanodegree on Udacity

## Summary
This dataset contains data about direct marketing campaigns of a Portuguese banking institution based on phone calls. We seek to predict whether the client will subscribe the product (a bank term deposit).
(https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv)

Original dataset available on (https://www.kaggle.com/henriqueyamahata/bank-marketing)

The best performing model was a VotingEnsemble with an accuracy of 0.91681.


## Scikit-learn Pipeline

Main component of the pipeline is the Python script `train.py`, which performs the following steps:
1. Loading the dataset
2. Cleaning the data by removing missing values, encoding fields into categories, transforming text into numbers
3. Perform SKlearn Logistic Regression defined by two parameters:
   * *--C* (float): Inverse of regularization strength
   * *max_iter* (int): Maximum number of iterations taken for the solvers to converge

The Scikit-learn Pipeline is built in a Notebook:
1. Initialize a workspace and an experiment
2. Create a compute cluster
3. Define Hyperparameters including a policy for early termination and an estimator for the `train.py` script
4. Perform the Logistic Regression by defining the parameters (`--C` and `max_iter`) in HyperDrive and runnign the Python script `train.py`
5. Retrieve the metrics for the best model and save the model

### Data

Input variables:
* *age* (numeric)
* *job* : type of job (categorical)
* *marital* : marital status (categorical)
* *education* (categorical)
* *default*: has credit in default? (categorical)
* *housing*: has housing loan? (categorical)
* *loan*: has personal loan? (categorical)
* *contact*: contact communication type (categorical)
* *month*: last contact month of year (categorical)
* *day_of_week*: last contact day of the week (categorical)
* *duration*: last contact duration, in seconds (numeric)
* *campaign*: number of contacts performed during this campaign and for this client (numeric)
* *pdays*: number of days that passed by after the client was last contacted from a previous campaign (numeric)
* *previous*: number of contacts performed before this campaign and for this client (numeric)
* *poutcome*: outcome of the previous marketing campaign (categorical)
* *emp.var.rate*: employment variation rate - quarterly indicator (numeric)
* *cons.price.idx*: consumer price index - monthly indicator (numeric)
* *cons.conf.idx*: consumer confidence index - monthly indicator (numeric)
* *euribor3m*: euribor 3 month rate - daily indicator (numeric)
* *nr.employed*: number of employees - quarterly indicator (numeric)

Output variable (target):
* *y*: has the client subscribed a term deposit? (binary: 'yes','no')

### Parameter Sampler
Random Parameter Sampling is much faster than Grid Parameter Sampling but also provides reasonable results.

### Early stopping policy
The Bandit Policy defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. (https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py)
This policy allows very offensive savings.


## AutoML
**AutoML** tries to find the best model by automatically running different models and algorithms. The best performing model was `Voting Ensemble`.

Ensemble learning improves machine learning results and predictive performance by combining multiple models as opposed to using single models. Voting Ensemble predicts based on the weighted average of predicted class probabilities (for classification tasks) or predicted regression targets (for regression tasks). (https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml#ensemble)

AutoML runs various iterations calculating different models. The Voting Ensemble model combines results of the following iterations / models:

| Ensembled Iteration | Ensembled Algorithm | Ensemble Weight |
| ------------------- |:-------------------:| ---------------:|
| 1 | XGBoostClassifier | 0.21428571428571427 |
| 0 | LightGBM | 0.42857142857142855 |
| 6 | XGBoostClassifier | 0.07142857142857142 |
| 14 | XGBoostClassifier | 0.14285714285714285 |
| 11 | XGBoostClassifier | 0.07142857142857142 |
| 9 | LogisticRegression | 0.07142857142857142 |

## Pipeline comparison
The best run of the Logistic Regression model tuned with HyperDrive provided an accuracy of *0.9073*

![alt text](https://github.com/HaslRepos/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/HyperDrive%20Top%2010.PNG "HyperDrive - Top 10 Results")

The best model identified by AutoML was a VotingEnsemble with an accuracy of *0.91681*

![alt text](https://github.com/HaslRepos/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/AutoML%20Top%2010.PNG "AutoML - Top 10 Results")

AutoML outperforms the HyperDrive model, although the difference is only marginal.
With regard to architecture both models pursuit a different approach. The HyperDrive pipeline performes experiments with different parameters based on a single model whereas AutoML examines different models and algorithms.


## Future work
Apply additional transformations (eg. Synthetic Minority Oversampling Technique) during data preparation to handle the imbalanced dataset. 89 % of the data is labled No and might result in biased predictions.
