# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import the necessary tensorflow modules

### STEP 2:
load the stock dataset

### STEP 3:
Fit the model and train the data

## STEP 4 :
predict the values
```
## PROGRAM


name : k.garshankumar
reg no: 212219040034
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

df_train = pd.read_csv('trainset.csv')
df_train.head(60)

train_set = df_train.iloc[:,1:2].values
train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)


X_train

X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train1

model = Sequential([layers.SimpleRNN(50,input_shape=(60,1)),
                    layers.Dense(1)
                    ])

model.compile(optimizer='Adam', loss='mae')

model.fit(X_train1,y_train,epochs=100,batch_size=32)

df_test=pd.read_csv("testset.csv")
test_set = df_test.iloc[:,1:2].values

dataset_total = pd.concat((df_train['Open'],df_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```

## OUTPUT
![image](https://user-images.githubusercontent.com/112486797/194876106-22231627-7bc7-4c9c-a1b3-4999b0e8c957.png)


![image](https://user-images.githubusercontent.com/112486797/194876272-03716ca5-6f8a-4805-9c09-7790ca6fb8cd.png)



## RESULT
A Recurrent Neural Network model for stock price prediction is developed.
