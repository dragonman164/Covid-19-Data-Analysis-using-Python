import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("covid19_italy_region.csv")
data.head()
data.columns
data.tail()
data.describe()
data.isnull().sum()


sns.relplot(x="TotalPositiveCases",y="Recovered",data=data)
sns.relplot(x="TotalPositiveCases",y="HospitalizedPatients",data=data)
sns.relplot(x="TotalPositiveCases",y="HospitalizedPatients",hue="Recovered",data=data)
sns.pairplot(data)
sns.relplot(x='TotalPositiveCases',y='HomeConfinement',kind='line',data=data)
sns.relplot(x='Recovered',y='HomeConfinement',kind='line',data=data)
sns.catplot(x='Recovered',y='TotalPositiveCases',data=data)