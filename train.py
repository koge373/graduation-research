from emg import (
        Preprocess, BuildCNN , plot_history, train,Preprocess_all
)
from predict import (
        pred
        )
import numpy as np

def bond(x_train,x_train1,y_train,y_train1):
    x_train = np.concatenate([x_train,x_train1], axis=0)
    y_train = np.concatenate([y_train,y_train1], axis=0)
    for l in [x_train, y_train]:
        np.random.seed(1)
        np.random.shuffle(l)
    return x_train,y_train
        
if __name__ == '__main__':
    #x_train, x_val, y_train, y_val =Preprocess_all()
    x_train, x_val, y_train, y_val =Preprocess(0,0)
    x_train1, x_val1, y_train1, y_val1=Preprocess(1,0.5)
    x_train , y_train = bond(x_train,x_train1,y_train,y_train1)
    model = BuildCNN()
    hist = train(model,x_train, x_val1, y_train, y_val1)
    #hist = train(model,x_train, x_val, y_train, y_val)
    plot_history(hist)
    pred(x_val1,y_val1)