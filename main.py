import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

df=pd.read_csv("insurance.csv")
df.isnull().sum()
lb=LabelEncoder()
df['sex']=lb.fit_transform(df['sex'])
df['smoker']=lb.fit_transform(df['smoker'])
df['region']=lb.fit_transform(df['region'])
print(df.head(2))

x=df.drop(columns=['charges'])
y=df['charges']


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_pred,y_test)
