import pandas as pd
import numpy as np
import os
import pickle


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('weatherhistory.csv')

print(data.head())

data_num=data[list (data.dtypes [data.dtypes!='object'].index)]


data_num = data_num.rename(columns={'Temperature (C)':'Temperature','Apparent Temperature (C)':'ApparentTemperature','Wind Speed (km/h)':'WindSpend','Wind Bearing (degrees)':'WindBearing','Visibility (km)':'Visibility','Loud Cover':'LoudCover','Pressure (millibars)':'Pressure'})

final=('Temperature')

weather_y = data_num.pop('Temperature')

weather_X = data_num

train_X,test_X, train_y,test_y = train_test_split(weather_X, weather_y, test_size = 0.2, random_state=4)

train_X.head()

train_y.head()

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=0)


model.fit(train_X, train_y)

# Saving model to disk
pickle.dump(model, open('modelpy.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('modelpy.pkl','rb'))

# print(model.predict([[2, 9, 6]]))