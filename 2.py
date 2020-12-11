import numpy as np 
import pandas as pd
import seaborn as sns

data = pd.read_csv("bmi.csv")

data.head()
data.tail()
data.describe()

from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

gender = LabelEncoder()
data['Gender'] = gender.fit_transform(data['Gender'])
data.head()

bins = (-1,0,1,2,3,4,5)
health = ['Malnourished','Underweight','fit','Slightly Overweight','Overwieght','Extremely Overweight']
data['Index']=pd.cut(data['Index'],bins=bins,labels=health)

data.head()

data['Index'].value_counts()
data['Gender'].value_counts()

sns.countplot(data['Gender'])

sns.countplot(data['Index'])

sns.relplot(x="Height",y="Weight",hue="Index",data=data)
sns.relplot(x="Index",y="Weight",hue="Gender",height=5,aspect=3,data=data)

sns.relplot(x="Index",y="Weight",hue="Gender",height=5,aspect=3,data=data,kind='line')

X = data.drop('Index',axis=1)
y = data['Index']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

clf = svm.SVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))

a = [[0,160,48]]
a = s.transform(a)
b = clf.predict(a)
