# Zoomcamp Midterm Project

This is the first project work after 7 weeks of ML ZoomCamp course by Alexey Grigorev which summarises the gained knowledge.
- https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp

## Table of Contents

[1. The project](#the-project)

[2. The files](#the-files)

[3. Deployment to AWS](#deployment-to-aws)


### The project
I have chosen the fraud prediction dataset because this problem is considered the biggest threat to a business by 36% of companies in today's e-commerce landscape. For example, 2021 was one of the most eventful years on record for identity theft and credit card fraud. 

## Theft identification in the United States

![image](https://user-images.githubusercontent.com/58089872/199707659-4b16d629-43e9-4703-ae9e-1c505bbc4641.png)

Retail fraud is an illegal transaction that a fraudster performs using stolen credit card details or loopholes in the order placement and payment systems and company policies. The first step to combat retail fraud is it's detection. To effectively prevent the criminal actions that lead to the leakage of bank account information should consider the implementation of advanced Credit Card Fraud Prevention and Fraud Detection methods. Machine Learning-based methods can continuously improve the accuracy of fraud prevention based on information about each cardholder’s behavior.

[Go back to table of content](#table-of-contents)
#### About Dataset: 

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

-The dataset contains all credit card transaction details.

- Except all columns class column denotes following:

  - Class 0 --> Non fraudulent

  - Class 1 --> fraudulent

- Goal: 

  - The main goal is to build various different algos.

  - Solving method:

The given problem statement is comes under binary classification
We have to solve problem using different machine learning algorithms

### The files
- Data
  
  - `creditcard.csv` - dataset used for this project

- Notebook 
  
  - `FraudDetection.ipynb` - contains EDA,  data preparation, Model selection process and parameter tuning

- Scripts

   - `train.py` - training the final model, saving it to a file 
   - `predict.py` - loading the model, serving it via a web serice (with Flask)
   - `Pipenv` and `Pipenv.lock` - files with dependencies
   - Dockerfile for running the service
   
[Go back to table of content](#table-of-contents)

### Deployment to AWS

<img width="1112" alt="image" src="https://user-images.githubusercontent.com/58089872/199556886-c579cd81-d879-4451-9372-aa1fc2f9cce9.png">

[Go back to table of content](#table-of-contents)
