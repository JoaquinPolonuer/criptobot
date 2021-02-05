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
        self.training_data_len = 0

        self.scaled_data = []
        self.dataset = []
        self.data = []

        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []     

        self.model = []   
        self.scaler = []        

    def load_dataset(self,dataframe, currency):
        self.df = dataframe
        self.currency = currency

    
    def prepare_dataset(self):
        self.data = self.df.filter(["close"])
        #convert dataframe to np array
        self.dataset = self.data.values
        self.training_data_len = math.ceil(len(self.dataset)*self.train_percentaje)
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_data = self.scaler.fit_transform(self.dataset) 
        train_data = self.scaled_data[0:self.training_data_len,:]

        self.x_train = []
        self.x_train = []

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

    def test_model(self):
        test_data = self.scaled_data[self.training_data_len - self.days_anticipated: , :]

        self.y_test = self.dataset[self.training_data_len: , :]

        self.x_test = []
        for i in range(self.days_anticipated, len(test_data)):
            self.x_test.append(test_data[i-self.days_anticipated:i, 0])
        
        self.x_test = np.array(self.x_test)

        self.x_test = np.reshape(self.x_test,(self.x_test.shape[0],self.x_test.shape[1],1))

        predictions = self.model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)

        rmse = np.sqrt(np.mean(((predictions- self.y_test)**2)))
        print(rmse)

        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:]
        valid["Predictions"] = predictions
        plt.figure(figsize=(16,8))
        plt.title("Model")
        plt.plot(train["close"])
        plt.plot(valid[["close", "Predictions"]])
        plt.legend(["Train", "Val", "Predictions"], loc="lower right")
        plt.show()

