import numpy as np


def main() :
    emg_list =[]
    for n in range(1,5) :
        file_name = 'emg%d.csv' % (n)
        data = np.loadtxt(file_name,delimiter=",",skiprows = 1)
        emg_list.append(np.split(data,45))
    print(np.array(emg_list).shape)

if __name__ == '__main__':
    main()
    