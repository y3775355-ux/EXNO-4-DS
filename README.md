# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
import pandas as pd

import numpy as np

df=pd.read_csv("D:bmi.csv")


<img width="486" height="626" alt="image" src="https://github.com/user-attachments/assets/2a89576c-ffe5-42a8-acc2-6e1b7a5341b9" />

df.dropna() 
<img width="456" height="636" alt="image" src="https://github.com/user-attachments/assets/e1ae0ff3-1747-4319-aa2f-b30f58a081
max_vals=np.max(np.abs(df[['Height','Weight']]))
f3" />

max_vals

<img width="960" height="53" alt="image" src="https://github.com/user-attachments/assets/5fcb323b-782d-4a10-bbfd-b9fbe59930a9" />

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df.head(10)

<img width="456" height="556" alt="image" src="https://github.com/user-attachments/assets/8b1ce7f3-fcfd-47cb-bb70-35626e0e6314" />

df1=pd.read_csv("bmi.csv")

df2=pd.read_csv("bmi.csv")

df3=pd.read_csv("bmi.csv")

df4=pd.read_csv("bmi.csv")

df5=pd.read_csv("bmi.csv")

df1
<img width="506" height="644" alt="image" src="https://github.com/user-attachments/assets/42e13ce1-1dd4-4cc7-98f9-8381877006d8" />

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])

df.head(10)

<img width="464" height="543" alt="image" src="https://github.com/user-attachments/assets/23dbe22e-e4e1-4933-a606-406dc696a14b" />

from sklearn.preprocessing import Normalizer

scaler=Normalizer()

df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])

df2

<img width="492" height="636" alt="image" src="https://github.com/user-attachments/assets/013cacc8-f4c6-440b-9be3-ba24a7925eb1" />

from sklearn.preprocessing import MaxAbsScaler

max1=MaxAbsScaler()

df3[['Height','Weight']]=max1.fit_transform(df3[['Height','Weight']])

df3

<img width="511" height="642" alt="image" src="https://github.com/user-attachments/assets/1df2e449-ccbf-4199-921c-387fa32c3ba2" />

from sklearn.preprocessing import RobustScaler

roub=RobustScaler()

df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])

df4
<img width="498" height="627" alt="image" src="https://github.com/user-attachments/assets/e8e63d92-3534-414e-873d-46394e7fe4aa" />

from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif

from sklearn.feature_selection import chi2

data=pd.read_csv("income(1) (1).csv")

data

<img width="996" height="290" alt="image" src="https://github.com/user-attachments/assets/d49fc1bb-c872-404f-8300-42862094bc3d" />

data1=pd.read_csv('/content/titanic_dataset (1).csv')

data1
<img width="868" height="316" alt="image" src="https://github.com/user-attachments/assets/b58c250f-cbde-4b26-96bc-45501af9b64b" />

data1=data1.dropna()

x=data1.drop(['Survived','Name','Ticket'],axis=1)

y=data1['Survived']

data1['Sex']=data1['Sex'].astype('category')

data1['Cabin']=data1['Cabin'].astype('category')

data1['Embarked']=data1['Embarked'].astype('category')

data1['Sex']=data1['Sex'].cat.codes

data1['Cabin']=data1['Cabin'].cat.codes

data1['Embarked']=data1['Embarked'].cat.codes

data1
<img width="886" height="329" alt="image" src="https://github.com/user-attachments/assets/199bec3f-45b6-41af-a5bf-4707259a8501" />

k=5

selector=SelectKBest(score_func=chi2,k=k)

x=pd.get_dummies(x)

x_new=selector.fit_transform(x,y)

x_encoded=pd.get_dummies(x)

selector=SelectKBest(score_func=chi2,k=5)

x_new=selector.fit_transform(x_encoded,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)

<img width="886" height="56" alt="Screenshot 2025-10-17 230709" src="https://github.com/user-attachments/assets/1b98a157-d80c-49ae-bf97-9f7892792ddd" />

 from sklearn.feature_selection import SelectKBest,f_regression
 
 import pandas as pd
 
 selector=SelectKBest(score_func=f_regression,k=5)
 
 x_new=selector.fit_transform(x_encoded,y)
 
 selected_feature_indices=selector.get_support(indices=True)
 
 selected_features=x.columns[selected_feature_indices]
 
 print("Selected Features:")
 
 print(selected_features)

 <img width="879" height="64" alt="image" src="https://github.com/user-attachments/assets/682858ed-e31a-4241-b848-3056608b6b8c" />

 from sklearn.feature_selection import SelectKBest,mutual_info_classif
 
 import pandas as pd
 
 selector=SelectKBest(score_func=mutual_info_classif,k=5)
 
 x_new=selector.fit_transform(x,y)
 
 selected_feature_indices=selector.get_support(indices=True)
 
 selected_features=x.columns[selected_feature_indices]
 
 print("Selected Features:")
 
 print(selected_features)

 <img width="882" height="69" alt="image" src="https://github.com/user-attachments/assets/c982fe0c-9b1a-41fa-9384-bc116ab86d2b" />

 from sklearn.feature_selection import SelectFromModel
 
 from sklearn.ensemble import RandomForestClassifier
 
 model=RandomForestClassifier()
 
 sfm=SelectFromModel(model,threshold='mean')
 
 x=pd.get_dummies(x)
 
 sfm.fit(x,y)
 
selected_features=x.columns[sfm.get_support()]

 print("Selected Features:")
 
 print(selected_features)

<img width="850" height="118" alt="image" src="https://github.com/user-attachments/assets/20e5b593-9c82-4221-98c7-c8d7199f9ccb" />

 from sklearn.ensemble import RandomForestClassifier
 
 model=RandomForestClassifier(n_estimators=100,random_state=42)
 
 model.fit(x,y)
 
 feature_selection=model.feature_importances_ threshold=0.1
 
 selected_features=x.columns[feature_selection>threshold]
 
 print("Selected Features:")
 
 print(selected_features)


 <img width="810" height="65" alt="image" src="https://github.com/user-attachments/assets/3c72acd5-ab13-4bbd-bfae-fafb6c855dde" />


 model=RandomForestClassifier(n_estimators=100,random_state=42)
 
 model.fit(x,y)
 
 feature_importance=model.feature_importances_ threshold=0.15
 
 selected_features=x.columns[feature_importance>threshold]
 
 print("Selected Features:")
 
 print(selected_features)


 <img width="458" height="76" alt="image" src="https://github.com/user-attachments/assets/4aabcc7f-764e-474a-875e-5247492e5575" />

# RESULT:
       thus  performed Feature Scaling and Feature Selection process and save the
data to a file.

# RESULT:
       # INCLUDE YOUR RESULT HERE
