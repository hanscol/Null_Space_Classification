import pickle
import os
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_img(folder, pickle_file):
    data = unpickle(pickle_file)
    keys = list(data.keys())

    data[keys[2]] = np.reshape(data[keys[2]], [10000, 3, 32, 32])
    data[keys[2]] = np.transpose(data[keys[2]], (0, 2, 3, 1))

    for i in range(data[keys[2]].shape[0]):
        img = data[keys[2]][i, :, :, :]
        label = data[keys[1]][i]
        name = data[keys[3]][i].decode('utf-8')
        io.imsave(os.path.join(folder, str(label), name), img)

path = '/home/local/VANDERBILT/hansencb/CIFAR-10/cifar-10-batches-py/'

files = os.listdir(path)

for file in files:
    if 'data' in file:
        folder = '/home/local/VANDERBILT/hansencb/CIFAR-10/Train/'
        save_img(folder, os.path.join(path, file))
    elif 'test' in file:
        folder = '/home/local/VANDERBILT/hansencb/CIFAR-10/Test/'
        save_img(folder, os.path.join(path, file))
