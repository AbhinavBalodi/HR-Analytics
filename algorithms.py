import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 

Emp_Perf = pd.read_excel('hrr1.xlsx')
x=Emp_Perf.iloc[:,0:25].values
y=Emp_Perf.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
x[:,1]=lab.fit_transform(x[:,1])
x[:,2]=lab.fit_transform(x[:,2])
x[:,3]=lab.fit_transform(x[:,3])
x[:,4]=lab.fit_transform(x[:,4])
x[:,5]=lab.fit_transform(x[:,5])
x[:,14]=lab.fit_transform(x[:,14])
x[:,24]=lab.fit_transform(x[:,24])
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder(categorical_features=[1,2,3,4,5,14,24])
x=one.fit_transform(x)
x=x.toarray()
s=pd.DataFrame(x)
x2=s.iloc[:,24:42].values
rr=pd.DataFrame(x2)
from sklearn.model_selection import train_test_split
x2_train, x2_test,y_train,y_test=train_test_split(x2,y ,test_size=0.25)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500,n_jobs=2,oob_score=True)
clf.fit(x2_train,y_train)
preds=clf.predict(x2_test)
y_pred=clf.predict(x2_test)
clf.score(x2_test,y_test)
from sklearn.tree import DecisionTreeClassifier
dtc =DecisionTreeClassifier(criterion="entropy",max_depth=8,min_samples_split=50,random_state=0)
dtc.fit(x2_train,y_train)
preds=dtc.predict(x2_test)
y_pred=dtc.predict(x2_test)
dtc.score(x2_test,y_test)
