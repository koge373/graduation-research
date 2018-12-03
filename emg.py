import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.utils import np_utils
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def main() :
    emg_list = []
    for n in range(1,6) :
        file_name = 'emg%d.csv' % (n)
        data = np.loadtxt(file_name,delimiter=",",skiprows = 1, dtype = np.float32)
        emg_list.extend(np.split(data,1000))
    emg_list = np.array(emg_list)/125.0
    #print(np.array(emg_list).shape)
    #print(emg_list)
    
    out_put = np.zeros(5000)
    for n in range (5) :
        out_put[n*1000:n*1000+1000] = n
    out_put = np_utils.to_categorical(out_put)  
    #print(out_put.shape)
    
    emg_list = emg_list.reshape(5000, 45, 8, 1)
    (x_train, x_val, y_train, y_val) = train_test_split(emg_list, out_put, test_size=0.2)

    
    model = Sequential()
    model.add(Conv2D(24, 3, padding='same', input_shape=(45,8,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(48, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(96, 3, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(96, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    
    mcp = ModelCheckpoint(filepath='best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    print(">> 学習開始")
    hist = model.fit(x_train, y_train,
                     batch_size=32,
                     verbose=1,
                     epochs=50000,
                     validation_data=[x_val, y_val],
                     callbacks=[mcp])

if __name__ == '__main__':
    main()
    