from emg import (
        Preprocess , BuildCNN , plot_history, train
)

if __name__ == '__main__':
    x_train, x_val, y_train, y_val =Preprocess()
    model = BuildCNN()
    hist = train(model,x_train, x_val, y_train, y_val)
    plot_history(hist)
    
