import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
x=pd.read_csv("train.csv")

y=x.loc[:,"Loan_Status"]
x=x.drop("Loan_Status",axis=1)
x=x.drop("Loan_ID",axis=1)

#import test dataset
x_test=pd.read_csv("test.csv")
x_test
indexTest=x_test.loc[:,"Loan_ID"]
x_test=x_test.drop("Loan_ID",axis=1)
dataset=[x,x_test]
#mapping gender variable
gender_mapping={"Male":0,"Female":1}
for ds in dataset:
    ds["Gender"]=ds["Gender"].map(gender_mapping)
    #replacing Nan in gender
for ds in dataset:
    ds["Gender"].fillna(ds["Gender"].mean(),inplace=True)
#mapping Married variable
married_mapping={"Yes":0,"No":1}
for ds in dataset:
    ds["Married"]=ds["Married"].map(married_mapping)
#replacing Nan in married
for ds in dataset:
    ds["Married"].fillna(ds["Married"].mean(),inplace=True)
#encoding education
edu_mapping={"Graduate":0,"Not Graduate":1}
for ds in dataset:
    ds["Education"]=ds["Education"].map(edu_mapping)
#encoding dependents
dep_mapping={"1":1,"2":2,"0":0,"3+":3}
for ds in dataset:
    ds["Dependents"]=ds["Dependents"].map(dep_mapping)
#fill Nan of dependents
for ds in dataset:
    ds["Dependents"].fillna(ds["Dependents"].mean(),inplace=True)
#encoding self employed
self_mapping={"No":0,"Yes":1}
for ds in dataset:
    ds["Self_Employed"]=ds["Self_Employed"].map(self_mapping)

#fill nan of self employed
for ds in dataset:
    ds["Self_Employed"].fillna(ds.groupby("Education")["Self_Employed"].transform("median"),inplace=True)
#fill nan in LoanAmount
for ds in dataset:
     ds["LoanAmount"].fillna(ds.groupby("Education")["LoanAmount"].transform("median"),inplace=True)
#fill nan in LoanAmountterm
for ds in dataset:
     ds["Loan_Amount_Term"].fillna(ds.groupby("Education")["Loan_Amount_Term"].transform("median"),inplace=True)
#fill nan in credithistory
for ds in dataset:
     ds["Credit_History"].fillna(ds["Credit_History"].mean(),inplace=True)
#property area 
prop_mapping={"Urban":0,"Rural":1,"Semiurban":2}
for ds in dataset:
    ds["Property_Area"]=ds["Property_Area"].map(prop_mapping)
    
#train test split
from sklearn.model_selection import train_test_split
x_train,xx_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(xx_test)

from sklearn.neighbors import KNeighborsClassifier
classifier1=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier1.fit(x_train,y_train)
y_pred1=classifier1.predict(xx_test)

from sklearn.naive_bayes import GaussianNB
classifier2=GaussianNB()
classifier2.fit(x_train,y_train)
y_pred2=classifier2.predict(xx_test)


from sklearn.svm import SVC
classifier3=SVC()
classifier3.fit(x_train,y_train)
y_pred3=classifier3.predict(xx_test)



from sklearn.metrics import accuracy_score,confusion_matrix
a=accuracy_score(y_test,y_pred)
a1=accuracy_score(y_test,y_pred1)
a2=accuracy_score(y_test,y_pred2)
a3=accuracy_score(y_test,y_pred3)
cm=confusion_matrix(y_test,y_pred)
y_realpred=classifier.predict(x_test)
df=pd.DataFrame({"Loan_ID":indexTest,"Loan_Status":y_realpred})
df.to_csv("sample_submission1.csv",index=False)     
