import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt

################################
######### データの前処理 ##########
################################

def Preprocess() :
    emg_list = []
    for n in range(1,16) :
        file_name = 'emg_data/emg%d.csv' % (n)
        data = np.loadtxt(file_name,delimiter=",",skiprows = 1, dtype = np.float32)
        emg_list.extend(np.split(data,1000)) #45個で１塊のデータを1000個×8極
    emg_list = np.array(emg_list)/125.0 #値を小数に
   # print(np.array(emg_list).shape)
    #print(emg_list)
    
    out_put = np.zeros(15000)
    for n in range (5) :
        out_put[n*1000:n*1000+1000] = n
    for n in range(6,11) :
        out_put[n*1000:n*1000+1000] = n-6
    for n in range(11,16) :
        out_put[n*1000:n*1000+1000] = n-11


    out_put = np_utils.to_categorical(out_put)  # 自然数をベクトルに変換
   # print(out_put.shape)
    
    emg_list = emg_list.reshape(15000, 45, 8, 1)
    (x_train, x_val, y_train, y_val) = train_test_split(emg_list, out_put, test_size=0.2)
    
    return x_train, x_val, y_train, y_val

################################
########## モデルの構築 ###########
################################
    
def BuildCNN(ipshape=(45,8,1), num_classes=5) :
    model = Sequential()
    model.add(Conv2D(24, 3, padding='same', input_shape=ipshape))
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
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model
    
def train(model,x_train, x_val, y_train, y_val) :
    mcp = ModelCheckpoint(filepath='best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    print(">> 学習開始")
    history = model.fit(x_train, y_train,
                     batch_size=32,
                     verbose=1,
                     epochs=100,
                     validation_data=[x_val, y_val],
                     callbacks=[mcp])
            ### add to show graph
    return history

################################
########## グラフの作成 ###########
################################   
    
def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

def encode(data):
     data = np_utils.to_categorical(data) 
     return data
    
    
