#@title Default title text

#importing Tools to load and analyize the data
import pandas as pd
import seaborn as se
import matplotlib.pyplot as plt
import numpy as np
#loading the dataset csv file
dataset=pd.read_csv(r"/content/healthcare-dataset-stroke-data.csv")
print(dataset.head())#first five columns of dataset loaded
#check if any null values present in dataframe
print(dataset.isnull().sum())
dataset.drop("id",axis=1,inplace=True)#removing unwanted columns
meanvalues=dataset["bmi"].mean()#replace null values by mean value 
dataset["bmi"].fillna(meanvalues,inplace=True)
#below line represents the diffrent statistics values in dataframe
print(dataset.describe())
#Analyze the dataframe 
print(se.heatmap(data=dataset.corr(),annot=True))
# encoding the different categories values like "gender","ever_married","worktype,"smoking status".
dataset["gender"].unique()
def alter(col1):
  if col1=="Male":
    return 0
  elif col1=="Female":
    return 1
  else:
    return 2
dataset["gender"]=dataset["gender"].apply(alter) 
def change(col2):
  if col2=="Yes":
    return 1
  else:
    return 0  
dataset["ever_married"]=dataset["ever_married"].apply(change)    
def alter(col3):
    if col3 == 'Private':
        return 0
    elif col3 == 'Self-employed':
        return 1
    elif col3 == 'Govt_job':
        return 2
    elif col3 == 'children':
        return 3
    else:
        return 4

dataset['work_type'] = dataset['work_type'].apply(alter)
def change1(col4):
    if col4 == 'formerly smoked':
        return 0
    elif col4 == 'never smoked':
        return 1
    elif col4 == 'smokes':
        return 2
    else:
        return 3

dataset['smoking_status'] = dataset['smoking_status'].apply(change1)
def change(col5):
  if col5=="Urban":
    return 1
  else:
    return 0  
dataset["Residence_type"]=dataset["Residence_type"].apply(change) 
#converting the gender flot values into nearest integer both "gender" and "bmi" 
dataset["gender"]=dataset["gender"].apply(np.ceil).astype("Int64")
dataset["bmi"]=dataset["bmi"].apply(np.ceil).astype("Int64")
#Extracting the both dependent and independent values
x=dataset.iloc[:,[0,1,2,3,4,5,6]].values#indpendent values
y=dataset.iloc[:,-1].values#dependent values

#splitting the data into testing and training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#importing the standardscalar
from sklearn.preprocessing import StandardScaler  
scale=StandardScaler()
x_train=scale.fit_transform(x_train)#training the dataset
x_test=scale.transform(x_test)#testing the dataset
#selecting the suitable model for dataset
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
classifier.fit(x_train,y_train)
#now prediction of data
prediction=classifier.predict(x_test)
#check the accuracy of data after testing and training performed on dataset
from sklearn.metrics import confusion_matrix,accuracy_score
print("Accuracy_score:",accuracy_score(y_test,prediction)*100)
print("confusion_matrix:",confusion_matrix(y_test,prediction))


