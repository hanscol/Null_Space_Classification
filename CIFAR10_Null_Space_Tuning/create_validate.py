import os
import random

train_folder = '/home/local/VANDERBILT/hansencb/CIFAR-10/Train/'
test_folder = '/home/local/VANDERBILT/hansencb/CIFAR-10/Validate/'

class_folders = os.listdir(train_folder)
data = []
for c in class_folders:
    files = os.listdir(os.path.join(train_folder, c))

    for f in files:
        data.append(os.path.join(c,f))


random.shuffle(data)
for i in range(5000):
    os.rename(os.path.join(train_folder,data[i]), os.path.join(test_folder, data[i]))
