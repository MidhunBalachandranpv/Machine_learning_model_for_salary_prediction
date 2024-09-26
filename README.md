# Machine_learning_model_for_salary_prediction
## Objective
- To build a Machine learning model to predict the salary of a new employee joining the company
## Software used
- The entire Project was built using jupyter_notebook
## Language used
- The machine learning model was build using Python language
## Problem Statement
-  the problem statement tells us to create a machine learning algorithm to predict the salary of an new employee, the problem is quite challenging but interesting to do.
## Approach toward the Problem
 we have 3 given datas one is ML case study second one is cities and third one is colleges. my first step was to load these datas into jupyter notebook. after loading the data while checking the info i found out that there are 3 categorical variables in the data namely,colleges, cities and role and we also have colleges data in which all the colleges are classified into tier1,tier2 and tier3 also we have cities data in which all the cities are classified into metro and non metro cities. so my next task was to replace the college names in mL case study data with the tier1,tier2,tier3 values given in college data and also replace city names in ML case study with the metro or non_metro city to 1 and 0 as per given in the cities data. i have done the same. afterwards in role column we have 2 roles namely executive and manager, i have created dummy variable for that column in which for managerpost 1 is given and for executive 0 is given. the data dosen't had any missing values so i didin't had to do missing value imputation, afterwards i found out there are small outliers in the given data especially in the previous ctc aswell as ctc column thus that outliers was treated using capping and flooring method.. after that since it is a regression problem i have created all the available regression model and selected one with highest accuracy the models i created were multiple linear regression, Ridge regrssion, decision tree, random forest ,bagging, boost.
 ### Step by step Method and Codes For the Machine_Learning_Model to predict salary of employee
 #### Data Pre-processing
 First of all we have imported all the necessary libraries for our analysis * numpy for numerical calculations * pandas for data manipulations * seaborn and matplotlib for visualization
 ```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
Now we have to load the ML case study data (dataset of the project) as df
```
df = pd.read_csv("C:/Users/pc/techworksconsulting/ML case Study.csv",header=0)
```
Now to view the loaded data we can use the code below
```
df.head()
```
Now we need to load the colleges data 
```
df_cl = pd.read_csv("C:/Users/pc/techworksconsulting/Colleges.csv",header=0)
```
We can view the loaded data by
```
df_cl.head()
```





