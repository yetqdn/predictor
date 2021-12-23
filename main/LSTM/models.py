from django.db import models

# Create your models here.
import math
from django.http import request
#import  pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import datetime
#from .views import get_stock

def lstm(namestock):
      df = pd.read_csv('D:\c++\dev1\yetq\predictor\main\datas\excel_'+namestock.lower()+'.csv')
      #crete a new dataframe with only close column
      data = df.filter(['<CloseFixed>'])

      #convert the dataframe to a numpy array
      dataset=data.values
      #get the number of rows to train the model on
      training_data_len = math.ceil(len(dataset)*.8)
      #scale the data
      scaler = MinMaxScaler(feature_range=(0,1))
      scaled_data = scaler.fit_transform(dataset)
      # create the scaled training data set
      train_data = scaled_data[0:training_data_len, :]
      #split the data into x-train and y_train data sets
      x_train = []
      y_train = []
      for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
      #convert the x_train and y-train to numpy arrays
      x_train, y_train = np.array(x_train), np.array(y_train)
      #reshape the data
      x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
      #Build the LSTM model
      model = Sequential()
      model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
      model.add(LSTM(50, return_sequences=False))
      model.add(Dense(25))
      model.add(Dense(1))
      #compile the model
      model.compile(optimizer='adam', loss='mean_squared_error')

      #train the model
      model.fit(x_train, y_train, batch_size=1, epochs=1)

      #create the testing data set
      #create a new array containing scaled values from index to 2023
      test_data = scaled_data[training_data_len-60: , :]
      #create the data sets x_test and y_test
      x_test = []
      y_test = dataset[training_data_len:, :]
      for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i,0])
      #convert the data to a numpy array
      x_test = np.array((x_test))
      #reshape the data
      x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
      #Get the models predicted price values
      predictions = model.predict(x_test)
      predictions = scaler.inverse_transform(predictions)
      #get the root squared error (RMSE)
      rmse = np.sqrt(np.mean(predictions - y_test)**2)
      #plot data
      train = data[:training_data_len]
      valid = data[training_data_len:]
      valid['Predictions'] = predictions
      return df['<DTYYYYMMDD>'], df['<CloseFixed>'], valid['Predictions']
      """time = df['<DTYYYYMMDD>']
      close = df['<CloseFixed>']
      predicted = valid['Predictions']"""
            
     

#get the data and show it (index_col=1)

"""

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['<CloseFixed>'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price VND', fontsize=18)
plt.show()

#crete a new dataframe with only close column
data = df.filter(['<CloseFixed>'])

#convert the dataframe to a numpy array
dataset=data.values
#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*.8)
#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
# create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
#split the data into x-train and y_train data sets
x_train = []
y_train = []
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])
#convert the x_train and y-train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
#model.fit(x_train, y_train, batch_size=1, epochs=1)

#create the testing data set
#create a new array containing scaled values from index to 2023
test_data = scaled_data[training_data_len-60: , :]
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i,0])
#convert the data to a numpy array
x_test = np.array((x_test))
#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
#get the root squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
#plot data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close', fontsize=18)
plt.plot(train['<CloseFixed>'])
plt.plot(df['<CloseFixed>'])
plt.plot(valid[ 'Predictions'])
plt.legend(['Train', 'df', 'Predictions'], loc='upper right')
#plt.show()
"""