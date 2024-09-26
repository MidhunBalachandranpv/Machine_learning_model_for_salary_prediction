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
We need to create a list of tire1 colleges by dropping na values for that use
```
Tier1 = df_cl["Tier 1"].dropna().tolist()
```
To view columns of dataset use
```
print(df.columns)
```
Now we need to replace the name of colleges with 1,2,3 where it represents tier1,tier2 and tier3 respectively since it is a categorical value
```
for item in df["College"]:
    if item in Tier1:
        df["College"].replace(item, 1, inplace=True)
    elif item in Tier2:
        df["College"].replace(item, 2, inplace=True)
    elif item in Tier3:
        df["College"].replace(item, 3, inplace=True)
```
To View first 10 raws of edited dataset use 
```
df.head(10)
```
Now we need to load cities.csv file for that use
```
df_cty = pd.read_csv("C:/Users/pc/techworksconsulting/cities.csv",header=0)
```
Now wee need to create a list of metro cities
```
metro_city = df_cty["Metrio City"].dropna().tolist()
```
Similarly we need to create a list for non metro cities
```
non_metro_city = df_cty["non-metro cities"].dropna().tolist()
```
Now we need to replace the value of metrocities with 1 and non metro cities with 0
```
for item in df["City"]:
    if item in metro_city:
        df["City"].replace(item, 1, inplace=True)
    elif item in non_metro_city:
        df["City"].replace(item, 0, inplace=True)
   ```
Now all our data is in required form all numericals and no categorical values

Now we have to preprocess our data further, we have transformed our categorical variable into a numerical and now we have to :
- look for outliers
- missing value imputation
    use this code to look for missing values
  
    ```
    df.info()
    ```
    As we can see from the above info there are no missing values, but we have an object named role so we need to assign dummy variable to that so that we can proceed further.. to do that
    ```
    df = pd.get_dummies(df,columns=["Role"],drop_first=True,dtype= int)
    ```
    Recheck the info by
    ```
    df.info()
    ```
    we have created dummy variables and we didin't had any missing values no we have to check whether the data have any outliers...
    ```
    sns.boxplot(x= "Previous CTC",data=df)
    sns.boxplot(x= "Previous job change",data=df)
    sns.boxplot(x= "Graduation Marks",data=df)
    sns.boxplot(x= "EXP (Month)",data=df)
    sns.boxplot(x= "CTC",data=df)
    ```
    As we can see our previous ctc and ctc contains a very little outlier and all the other independent variables does not have any outliers so let us treat the outlier present in previous ctc and ctc. to treat outliers use
    ```
    uv = np.percentile(df['Previous CTC'], 99)
    ```
    ```
    df[(df['Previous CTC'] > uv)]
    ```
    ```
    df['Previous CTC'][(df['Previous CTC'] > 3*uv)] = 3*uv
    ```
    ```
    uv1 = np.percentile(df['CTC'], 99)
    ```
    ```
    df[(df['CTC'] > uv1)]
    ```
    ```
    df['CTC'][(df['CTC'] > 3*uv1)] = 3*uv1
    ```
   We have treated outliers in variables by capping and flooring method now we have treated all outliers, we have done missing value imputation and also assigned dummy values for categorical variables thus our data preprocessing part is complete..

Now We have to split our data into independent and dependent variable where X is the independent variables and Y is the dependent variable
```
X= df.loc[:,df.columns != "CTC"]
Y= df["CTC"]
```
Now we have to do test train-split of data where 80% of data will be train data and 20% of data will be test data.. for that import sklearn library

```
from sklearn.model_selection import train_test_split
```
```
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
```
We have succesfully splitted data into test and train data where 80% of data is train data and 20% is test data. Now we can try different ML algorithms and check accuracy to find which model suits for our model

### Multiple Linear Regression Model
```
from sklearn.linear_model import LinearRegression
```
```
lm = LinearRegression()
lm.fit(X_train,Y_train)
```
We have created and fitted our training data in to the regression model\
```
Y_train_pred = lm.predict(X_train)
Y_test_pred = lm.predict(X_test)
Y_test_pred
```
We got the predicted y output on test data now let us see the r squared value and mse of this model to see the accuracy of our model
```
from sklearn.metrics import r2_score,mean_squared_error
```
```
r2_score (Y_train,Y_train_pred)
```
```
r2_score (Y_test,Y_test_pred)
```
```
mean_squared_error(Y_train,Y_train_pred)
```
```
mean_squared_error(Y_test,Y_test_pred)
```
### Ridge Regression model
```
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
from sklearn.linear_model import Ridge
lm_r = Ridge(alpha = 0.5)
lm_r.fit(X_train_s,Y_train)
r2_score(Y_test,lm_r.predict(X_test_s))
```
```
from sklearn.model_selection import validation_curve
```
```
param_range = np.logspace(-2,8,100)
```
```
train_scores, test_scores = validation_curve(
    Ridge(),  
    X_train_s, 
    Y_train,  
    param_name="alpha",  
    param_range=param_range, 
    scoring="r2",  
    cv=None  
)
```
```
print(train_scores)
print(test_scores)
```
```
train_mean = np.mean(train_scores,axis = 1)
```
```
test_mean = np.mean(test_scores,axis = 1)
```
```
max(test_mean)
```
```
sns.jointplot(x= np.log(param_range),y = test_mean)
```
```
np.where(test_mean == max(test_mean))
```
```
param_range[26]
```
```
lm_r_best = Ridge(alpha = param_range[26])
```
```
lm_r_best.fit(X_train_s,Y_train)
```
```
r2_score(Y_test,lm_r_best.predict(X_test_s))
```
We have also created and tested Ridge regression model by importing ridge from sklearn library
### Regression decision Tree
```
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 4)
regtree.fit(X_train,Y_train)
y_train_pred1 = regtree.predict(X_train)
y_test_pred1 = regtree.predict(X_test)
y_test_pred1
```
```
mean_squared_error(Y_test,y_test_pred1)
```
```
r2_score(Y_train,y_train_pred1)
```
```
r2_score(Y_test,y_test_pred1)
```
We have created and tested regression decision tree by importing tree from sklearn library



















   
    

















