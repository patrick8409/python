import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

data=pd.read_csv('banknote.csv')
dataset=np.loadtxt(data,delimiter=',')
x=data.iloc[:,0:4].values
y=data.iloc[:,4].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=17)
#Bernoulli
BernNB= BernoulliNB(binarize=True)
BernNB.fit(x_train,y_train)
print(BernNB)
y_expect=y_test
y_pred=BernNB.predict(x_test)
print(accuracy_score(y_expect,y_pred))
print("\n")
#gaussian

GausNB=GaussianNB()
GausNB.fit(x_train,y_train)
y_pred=GausNB.predict(x_test)
print(GausNB)
print(accuracy_score(y_expect,y_pred))
