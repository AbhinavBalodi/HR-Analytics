import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,TruncatedSVD
Emp_Perf = pd.read_excel('hrr1.xlsx')
print(Emp_Perf.columns)
print(Emp_Perf.shape)

# Drop the 'county_name' and 'state' columns
Emp_Perf.drop(['Age', 'Gender','EducationBackground',
'MaritalStatus',
'EmpDepartment',
'BusinessTravelFrequency',
'DistanceFromHome','EmpWorkLifeBalance',
'EmpRelationshipSatisfaction',
'EmpEducationLevel'], axis='columns', inplace=True)

# Examine the shape of the DataFrame (again)
print(Emp_Perf.shape)
print(Emp_Perf.columns)
x=Emp_Perf.iloc[:,0:15].values
y=Emp_Perf.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y ,test_size=0.25)
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
x_train[:,6]=lab.fit_transform(x_train[:,6])
x_train[:,14]=lab.fit_transform(x_train[:,14])


x_test[:,6]=lab.fit_transform(x_test[:,6])
x_test[:,14]=lab.fit_transform(x_test[:,14])

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500,n_jobs=2,oob_score=True)
clf.fit(x_train,y_train)
preds=clf.predict(x_test)
y_pred=clf.predict(x_test)
clf.score(x_test,y_test)
featImp=pd.DataFrame(data=clf.feature_importances_*100.0,columns=["GiniValue"])
feature_list=list(Emp_Perf.columns[0:15])
featImp.index=feature_list
featImp.sort_values(['GiniValue'],axis=0,ascending=False,inplace=True)
print(featImp.head())