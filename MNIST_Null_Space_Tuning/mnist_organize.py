import os
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from mnist import MNIST
import warnings
warnings.filterwarnings("ignore")

def save_img(folder, imgs, labels):

    imgs = np.array(imgs)
    imgs = np.reshape(imgs, [imgs.shape[0], 28, 28])

    for i in range(imgs.shape[0]):
        img = imgs[i, :, :]
        label = labels[i]
        name = '{:06d}.png'.format(i)
        io.imsave(os.path.join(folder, str(label), name), img)

path = '/home/hansencb/MNIST/byte_files/'
data = MNIST(path)

images, labels = data.load_training()
folder = '/home/hansencb/MNIST/Train/'

save_img(folder, images, labels)

images, labels = data.load_testing()
folder = '/home/hansencb/MNIST/Test/'

save_img(folder, images, labels)
