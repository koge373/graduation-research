import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from emg import (
        Preprocess , BuildCNN
)

def encode(vector):
    return np.round(vector)

def decode(data):
    data=np.argmax(data,axis=1)
    data=data.tolist()
   # print(data)
    return data

def emg_data(data_num,movement_num):
    emg_list = []
    file_name = 'emg_data/emg%d.csv' % (data_num)
    data = np.loadtxt(file_name,delimiter=",",skiprows = 1, dtype = np.float32)
    emg_list.extend(np.split(data,1000)) #45個で１塊のデータを1000個×8極
    emg_list = np.array(emg_list)/125.0 #値を小数に
    out_put = np.zeros(1000)
    out_put[0:1000] = movement_num
    out_put = np_utils.to_categorical(out_put)  # 自然数をベクトルに変換
    emg_list = emg_list.reshape(1000, 45, 8, 1)
    (x_train, x_val, y_train, y_val) = train_test_split(emg_list, out_put, test_size=0.5) 
    return x_train, x_val, y_train, y_val

def pred(x_val,y_val):
    model = BuildCNN()
    model.load_weights("best.hdf5")
    y_pred = model.predict( x_val, batch_size=None, verbose=1)
    y_pred = encode(y_pred)
    y_true=decode(y_val)
    y_pred=decode(y_pred)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print('\n')
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(confusion_matrix(y_true, y_pred,labels=[0,1,2,3,4]))
    
if __name__ == '__main__':
    model = BuildCNN()
    model.load_weights("best.hdf5")
    for n in range(16,18):
        x_train, x_val, y_train, y_val = emg_data(n,n-16)
        pred(x_val,y_val)
    model = BuildCNN()
    x_train, x_val, y_train, y_val = Preprocess(2,0.99)
    pred(x_val,y_val)



