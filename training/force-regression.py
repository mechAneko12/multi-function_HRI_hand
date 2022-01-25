from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
# from keras.utils import multi_gpu_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import r2_score
import pickle 

dataset_name = 'nakashima'
from pathlib import Path

path = Path(__file__).parent
path /= dataset_name + '/'


def load_dataset():
    # parting dataset
    data = pd.read_csv(path / 'forceM_train.csv')
    prices = data['f'].values
    prices = prices.reshape(-1, 1)
    # prices = prices / prices.max()
    features = data.drop('f', axis=1).values

    data_test = pd.read_csv(path / 'forceM_test.csv')
    pricest = data_test['f'].values
    pricest = pricest.reshape(-1, 1)
    # prices = prices / prices.max()
    featurest = data_test.drop('f', axis=1).values

    x_train = features
    y_train = prices
    x_valid = featurest
    y_valid = pricest
    return x_train, y_train, x_valid, y_valid

x_train, y_train, x_valid, y_valid = load_dataset()

print(x_train)
print(y_train)
print(x_valid)
print(y_valid)
y_true = y_valid

x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(8))
print('-------------------')
print(y_train_pd.head(8))

# generalization
min_max_scaler_x = MinMaxScaler()
min_max_scaler_x.fit(x_train_pd)
x_train = min_max_scaler_x.transform(x_train_pd)

min_max_scaler_y = MinMaxScaler()
min_max_scaler_y.fit(y_train_pd)
y_train = min_max_scaler_y.transform(y_train_pd)

# generalization
x_valid = min_max_scaler_x.transform(x_valid_pd)

y_valid = min_max_scaler_y.transform(y_valid_pd)


model = Sequential()
model.add(Dense(units = 10,
                activation='relu',
                input_shape=(x_train_pd.shape[1],)
               )
         )

model.add(Dropout(0.2))

model.add(Dense(units = 15,
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activity_regularizer=regularizers.l1(0.01),
                activation='relu' # Relu fuction
                # bias_regularizer=keras.regularizers.l1_l2(0.01)
               )
         )

model.add(Dense(units = 1,
                activation='linear'
               )
         )

print(model.summary())  # nn structure

model.compile(loss='mse',
              optimizer='adam',
             )

history = model.fit(x_train, y_train,
          epochs=200,  # iteration
          batch_size=200,  # batch sizw
          verbose=2,  # verbose：0：no output，1：ongoing，2：every epoch
          validation_data = (x_valid, y_valid)  # validation
        )


import matplotlib.pyplot as plt
# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# from keras.utils import plot_model
from keras.models import load_model
# save model
path_trained = path / 'trained_model'
model.save(path_trained / 'reg_middle.h5')  # creates a HDF5 file 'my_model.h5'

with open(path_trained / 'scaler_x_middle.pickle', mode='wb') as fp:
    pickle.dump(min_max_scaler_x, fp)
with open(path_trained / 'scaler_y_middle.pickle', mode='wb') as fp:
    pickle.dump(min_max_scaler_y, fp)



# load model
model = load_model(path_trained / 'reg_middle.h5')

# predict
y_new = model.predict(x_valid)

y_new = min_max_scaler_y.inverse_transform(y_new)
error = y_true-y_new
plt.plot(y_new)
plt.plot(y_true)
plt.legend(['predicted', 'true'], loc='upper left')
plt.show()
print(r2_score(y_true, y_new))
#print(y_new)
#print(error)
