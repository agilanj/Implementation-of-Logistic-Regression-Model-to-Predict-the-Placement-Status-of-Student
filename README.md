# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AGILAN J
RegisterNumber:  212224100002
```

```py

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

### Placement data:
<img width="1266" height="227" alt="image" src="https://github.com/user-attachments/assets/d11069e6-97a2-45ff-b0da-a771fae11250" />

### 2.Salary Data
<img width="1140" height="231" alt="image" src="https://github.com/user-attachments/assets/a5374eca-c1f6-4d83-8ae4-17ee4d6047a7" />


### 3. Checking the null function()
<img width="373" height="314" alt="image" src="https://github.com/user-attachments/assets/e05c4557-d574-4458-8d53-cc9926f0c398" />


### 4.Data Duplicate
<img width="228" height="43" alt="Screenshot 2026-01-29 153249" src="https://github.com/user-attachments/assets/64abf9cd-5381-4d0d-9a5d-b9515f6fc1d9" />


### 5.Print Data
<img width="1147" height="456" alt="image" src="https://github.com/user-attachments/assets/c8c485ca-edbd-4bac-9a1e-bdea3615191c" />


### 6.Data Status
<img width="643" height="268" alt="image" src="https://github.com/user-attachments/assets/d974bc46-9fb6-4881-a32d-3fe0b0af326d" />


### 7.y_prediction array
<img width="841" height="56" alt="image" src="https://github.com/user-attachments/assets/943c695e-116a-4da7-9c2b-7dde48193e2a" />


### 8.Accuracy value
<img width="322" height="50" alt="image" src="https://github.com/user-attachments/assets/9361ce45-8214-49d5-b531-0948c450eb05" />


### 9.Confusion matrix
<img width="461" height="69" alt="image" src="https://github.com/user-attachments/assets/0d837b6a-05f3-452d-b51b-fa78f0c9049e" />


### 10.Classification Report
<img width="628" height="208" alt="image" src="https://github.com/user-attachments/assets/d47755ff-7d2e-45c9-8493-00842e544aad" />


### 11.Prediction of LR
<img width="235" height="42" alt="image" src="https://github.com/user-attachments/assets/481dd0cb-494e-4b1e-a486-058cea30fa90" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
