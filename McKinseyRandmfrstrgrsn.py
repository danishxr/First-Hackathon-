
#lets load the basic data analysis packages 
import numpy as np
import pandas as pd 
import datetime as dt 

'''we are given train and test data we will train a model and improve it on training data 
and apply it to the test data to forcast the next 4 months vechicles at the 4 junctions'''

trdata=pd.read_csv('train_aWnotuB.csv')

#tsdata will be our forecast data 
tsdata=pd.read_csv('test_BdBKkAj.csv')

#data_wrangling -converting to time dtype then to ordinal values as regression takes ordinal values 
trdata['DateTime']=pd.to_datetime(trdata['DateTime'],errors='coerce')
trdata['Date']=trdata['DateTime'].dt.date 
trdata["Date"]=trdata["Date"].map(dt.datetime.toordinal)
trdata1=trdata.loc[:,['Junction','Vehicles','Date']].values


#same thing we take time dtype and convert to ordinal values
tsdata['DateTime']=pd.to_datetime(tsdata['DateTime'],errors='coerce')
tsdata['Date']=tsdata['DateTime'].dt.date 
tsdata["Date"]=tsdata["Date"].map(dt.datetime.toordinal)
tsdata11=tsdata.loc[:,['Junction','Date']].values

#encoding categorical variable here junctions 1,2,3,4
#one hot encoding it 
from sklearn.preprocessing import  OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
trdata1= onehotencoder.fit_transform(trdata1).toarray()
#same goes for our forecasting data 
tsdata11=onehotencoder.fit_transform(tsdata11).toarray()

#taking Training data in and setting dependent and independent variables 
x=trdata1[:,[0,1,2,3,5]]
y=trdata1[:,4]

#lets just split in the ratio 70:30
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

#we dont use feature scaling as the regression package does that automatically 



#We are going to use Random forest Regression.XGboost regreesion was tried but here Random Forest gave better result



from sklearn.ensemble import RandomForestRegressor
#now to choose the optimum parameter for Random forest 
'''from sklearn.model_selection import cross_val_score

j_shout=range(1,300)
j_acc=[]
for j in j_shout:  
   lr = RandomForestRegressor(n_estimators = j, random_state = 0)
   score=cross_val_score(lr,x_train,y_train,cv=10)
   j_acc.append(score.mean())


#executed 180 trees and lets find which gives the best testing accuracy 

f=j_shout[j_acc.index(max(j_acc))]'''


regressor = RandomForestRegressor(n_estimators = 270, random_state = 0)
regressor.fit(x_train, y_train)

    
# Predicting the Test set results
y_pred = regressor.predict(x_test)


#understanding the accuracy and standard deviation
from sklearn.model_selection import cross_val_score
accs=cross_val_score(regressor,X=x_train,y=y_train,cv=10)
accs.mean()
#0.77994
accs.std()
#0.01064

#Calculating the Root Mean Square Error
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))

#9.6353
 
#NOW LETS FORECAST FOR NEXT FOUR MONTHS USING OUR MODEL CREATED ON FORECASTING SET 

forcast_4months=regressor.predict(tsdata11)

M=tsdata['ID']
M=pd.DataFrame(M)
final_solution=pd.DataFrame(forcast_4months,columns=['Vehicles'])
M['Vehicles']=final_solution
M.to_csv('FINAL2_SOLUTION.csv')


#thank you!!!!!!
