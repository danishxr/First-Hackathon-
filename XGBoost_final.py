import numpy as np
import pandas as pd 
import datetime as dt 

#loading the training and test data into workspace 
trdata=pd.read_csv('train_aWnotuB.csv')
#tsdata=pd.read_csv('test_BdBKkAj.csv')
trdata['DateTime']=pd.to_datetime(trdata['DateTime'],errors='coerce')
trdata['Date']=trdata['DateTime'].dt.date 
trdata["Date"]=trdata["Date"].map(dt.datetime.toordinal)
trdata1=trdata.loc[:,['Junction','Vehicles','Date']].values
#trdata1= trdata1.replace(' ', '', regex=True)

'''tsdata['DateTime']=pd.to_datetime(tsdata['DateTime'],errors='coerce')
tsdata['Date']=tsdata['DateTime'].dt.date 
tsdata["Date"]=tsdata["Date"].map(dt.datetime.toordinal)
#encoding categorical variable her junctions 1,2,3,4
tsdata11=tsdata.loc[:,['Junction','Date']].values
tsdata11= tsdata11.replace(' ', '', regex=True)'''


#one hot encoding it 
from sklearn.preprocessing import  OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
trdata1= onehotencoder.fit_transform(trdata1).toarray()
x=trdata1[:,[0,1,2,3,5]]
y=trdata1[:,4]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

from xgboost import XGBRegressor
xregressor=XGBRegressor()
xregressor.fit(x_train,y_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
    
# Predicting the Test set results
y_pred = xregressor.predict(x_test)

from sklearn.model_selection import cross_val_score
accs=cross_val_score(xregressor,X=x_train,y=y_train,cv=10)
accs.mean()
accs.std()
regressor()

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))
 



