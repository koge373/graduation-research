from emg import (
        Preprocess , BuildCNN 
)

if __name__ == '__main__':
    x_train, x_val, y_train, y_val = Preprocess()
    model = BuildCNN()
    model.load_weights("best.hdf5")
    y_predict = model.predict( x_val, batch_size=None, verbose=1)
    print(y_val[0], y_predict[0])
    