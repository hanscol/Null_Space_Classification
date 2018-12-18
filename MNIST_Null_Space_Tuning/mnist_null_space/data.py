from __future__ import print_function, division
import torch
import torchvision

import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def train_files(folder):
    class_folders = os.listdir(folder)
    data = {}
    for c in class_folders:
        files = os.listdir(os.path.join(folder, c))

        for f in files:
            data[os.path.join(folder,c,f)] = int(c)

    return data

# def test_files(folder):
#     files = os.listdir(folder)
#     data = []
#     for f in files:
#         data.append(f)
#
#     return data

class Train_Dataset(Dataset):
    def __init__(self, data, null_split=1000):
        self.data = data
        self.keys = list(data.keys())
        self.null_split = null_split
        self.tensor = torchvision.transforms.ToTensor()

        self.class_to_files = {}
        for key in self.keys:
            label = self.data[key]
            if label not in self.class_to_files:
                self.class_to_files[label] = [key]
            else:
                self.class_to_files[label].append(key)

        for key in list(self.class_to_files.keys()):
            self.class_to_files[key] = sorted(self.class_to_files[key])

        if null_split > 0:
            class_split = int(null_split/10)
            diff = null_split - class_split*10
            for i in range(10):
                keys = self.class_to_files[i]
                if i == 9:
                    split_keys = keys[0:class_split+diff]
                else:
                    split_keys = keys[0:class_split]

                for key in split_keys:
                    del self.data[key]

                self.class_to_files[i] = split_keys

            self.keys = list(self.data.keys())


    def __len__(self):
        return max(len(self.keys), self.null_split)


    def preprocess(self, fname):
        img = io.imread(fname)
        img = img / 255.0
        transform.resize(img, [28, 28], mode='constant', anti_aliasing=True)

        img = np.expand_dims(img, axis=2)
        img = self.tensor(img)
        img = img.type(torch.float32)

        return img


    def __getitem__(self, idx):
        idx = idx % len(self.keys)
        fname = self.keys[idx]
        label = self.data[fname]

        img = self.preprocess(fname)

        null_label = random.randint(0, 9)
        null_keys = self.class_to_files[null_label].copy()
        idx = random.randint(0, len(null_keys) - 1)
        null_image1 = self.preprocess(null_keys[idx])
        del null_keys[idx]
        idx = random.randint(0, len(null_keys) - 1)
        null_image2 = self.preprocess(null_keys[idx])

        return {'image': img, \
                'target': torch.tensor(label), \
                'null_img1': null_image1,\
                'null_img2': null_image2 }


class Test_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.keys = list(data.keys())

        self.tensor = torchvision.transforms.ToTensor()


    def __len__(self):
        return len(self.keys)


    def preprocess(self, fname):
        img = io.imread(fname)
        img = img / 255.0
        transform.resize(img, [28, 28], mode='constant', anti_aliasing=True)

        img = np.expand_dims(img, axis=2)
        img = self.tensor(img)
        img = img.type(torch.float32)

        return img


    def __getitem__(self, idx):
        fname = self.keys[idx]
        label = self.data[fname]

        img = self.preprocess(fname)

        return {'image': img, \
                'target': torch.tensor(label), \
                'file': fname}
