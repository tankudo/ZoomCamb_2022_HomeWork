# Zoomcamp Midterm Project

This is the first project work after 7 weeks of ML ZoomCamp course by Alexey Grigorev which summarises the gained knowledge.
- https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp

## Table of Contents

[1. The project](#the-project)

[2. The files](#the-files)

[3. Deployment to AWS](#deployment-to-aws)


### The project
I have chosen the fraud prediction dataset because this problem is considered the biggest threat to a business by 36% of companies in today's e-commerce landscape. Retail fraud is an illegal transaction that a fraudster performs using stolen credit card details or loopholes in the order placement and payment systems and company policies. The first step to combat retail fraud is it's detection

[Go back to table of content](#table-of-contents)
#### About Dataset: 
-The dataset contains all credit card transaction details.

- Except all columns class column denotes following:

  - Class 0 --> Non fraudulent

  - Class 1 --> fraudulent

- Goal: 

  - The main goal is to build various different algos.

  - Solving method:

The given problem statement is comes under binary classification
We have to solve problem using different machine learning algorithm

### The files
- Data
  
  - `creditcard.csv` - dataset used for this project

- Notebook 
  
  - `FraudDetection.ipynb` - contains EDA,  data preparation, Model selection process and parameter tuning

- Scripts

   - 'train.py' - training the final model, saving it to a file 
   - `predict.py` - loading the model, serving it via a web serice (with Flask)
   - `Pipenv` and `Pipenv.lock` - files with dependencies
   - Dockerfile for running the service
   
[Go back to table of content](#table-of-contents)

### Deployment to AWS

<img width="1112" alt="image" src="https://user-images.githubusercontent.com/58089872/199556886-c579cd81-d879-4451-9372-aa1fc2f9cce9.png">

[Go back to table of content](#table-of-contents)
