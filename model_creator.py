import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


class Model_Creator:
    def __init__(self):
        self.df = []
        self.currency = ""   

        self.epochs = 1
        self.batch_size = 1
        self.days_anticipated = 60
        self.train_percentaje = 0.9

        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []     

        self.model = []           

    def load_dataset(self,dataframe, currency):
        self.df = dataframe
        self.currency = currency

    
    def prepare_dataset(self):
        data = self.df.filter(["close"])
        #convert dataframe to np array
        dataset = data.values
        training_data_len = math.ceil(len(dataset)*self.train_percentaje)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset) 
        train_data = scaled_data[0:training_data_len,:]
    
        for i in range(self.days_anticipated,len(train_data)):
            self.x_train.append(train_data[i-self.days_anticipated:i,0])
            self.y_train.append(train_data[i,0])
            # if i <= 60:
            #     print(x_train)
            #     print(y_train)
        
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1],1))

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.x_train.shape[1],1)))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def train_model(self):
        model_name = self.currency + "_" + str(self.epochs) +"_epochs_" + str(self.batch_size) + "_batch_" + str(self.days_anticipated) +"_days_anticipated"+".h5"
        print("will save at", model_name)
        self.model.fit(self.x_train,self.y_train,batch_size=self.batch_size,epochs=self.epochs)
        self.model.save(model_name)


