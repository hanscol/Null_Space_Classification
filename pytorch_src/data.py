from __future__ import print_function, division
import torch

from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import random
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class Train_Dataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.keys = list(data.keys())
        self.config = config
        self.null_split = int(config.null_split*len(self.keys))
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor = transforms.ToTensor()
        self.transforms = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.1, 0.1), shear=2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter()
        ])

        self.class_to_files = {}
        for key in self.keys:
            label = self.data[key]
            if label not in self.class_to_files:
                self.class_to_files[label] = [key]
            else:
                self.class_to_files[label].append(key)

        class_splits = []
        for key in range(len(self.class_to_files)):
            class_splits.append(int(self.null_split*(len(self.class_to_files[key])/len(self.keys))))

        self.null_class_to_files = {}
        if self.null_split > 0:
            diff = self.null_split - sum(class_splits)
            class_splits[np.argmax(class_splits)] += diff
            for i in range(len(class_splits)):
                class_split = class_splits[i]
                keys = self.class_to_files[i]
                split_keys = keys[0:class_split]

                self.null_class_to_files[i] = split_keys
                self.class_to_files[i] = keys[class_split:]

        self.len = 0
        if config.dataset == 'HAM10000':
            for i in range(len(self.class_to_files)):
                self.len = max(self.len, len(self.class_to_files[i]))
            self.len *= i
        else:
            for i in range(len(self.class_to_files)):
                self.len += len(self.class_to_files[i])

        if config.dataset == 'MNIST':
            self.len = max(self.len, self.null_split)

    def __len__(self):
        return self.len

    def preprocess(self, fname):
        img = io.imread(fname)
        if self.config.dataset == 'MNIST':
            img = img / 255.0
            transform.resize(img, [28, 28], mode='constant', anti_aliasing=True)

            img = np.expand_dims(img, axis=2)
            img = self.tensor(img)
            img = img.type(torch.float32)

        if self.config.dataset == 'CIFAR-10':
            img = img / 255.0
            img = transform.resize(img, [224, 224, 3], mode='constant', anti_aliasing=True)

            img = self.tensor(img)
            img = self.norm(img.float())
            img = img.type(torch.float32)

        if self.config.dataset == 'HAM10000':
            img = Image.fromarray(img.astype('uint8'))
            if self.transforms:
                img = self.transforms(img)

            img = np.array(img)
            img = img / 255.0
            img = transform.resize(img, [224, 224, 3], mode='constant', anti_aliasing=True)
            img = self.tensor(img)
            img = self.norm(img.float())
            img = img.type(torch.float32)

        return img

    def __getitem__(self, idx):
        label = random.randint(0, len(self.class_to_files)-1)
        idx = random.randint(0, len(self.class_to_files[label])-1)
        fname = self.class_to_files[label][idx]

        img = self.preprocess(fname)

        if self.config.null_space_tuning:
            null_label = random.randint(0, len(self.class_to_files)-1)
            null_keys = self.class_to_files[null_label].copy()
            idx = random.randint(0, len(null_keys) -1)
            null_image1 = self.preprocess(null_keys[idx])
            del null_keys[idx]
            idx = random.randint(0, len(null_keys) -1)
            null_image2 = self.preprocess(null_keys[idx])

            return {'image': img, \
                    'target': torch.tensor(label), \
                    'null_img1': null_image1,\
                    'null_img2': null_image2 }
        else:
            return {'image': img, \
                    'target': torch.tensor(label)}



class Test_Dataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.keys = list(data.keys())
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor = transforms.ToTensor()
        self.config = config

    def __len__(self):
        return len(self.keys)

    def preprocess(self, fname):
        img = io.imread(fname)
        img = img / 255.0

        if self.config.dataset == 'MNIST':
            transform.resize(img, [28, 28], mode='constant', anti_aliasing=True)
            img = np.expand_dims(img, axis=2)
            img = self.tensor(img)

        if self.config.dataset == 'CIFAR-10' or self.config.dataset == 'HAM10000':
            img = transform.resize(img, [224, 224, 3], mode='constant', anti_aliasing=True)
            img = self.tensor(img)
            img = self.norm(img.float())

        img = img.type(torch.float32)
        return img

    def __getitem__(self, idx):
        fname = self.keys[idx]
        label = self.data[fname]

        img = self.preprocess(fname)

        return {'image': img, \
                'target': torch.tensor(label), \
                'file': fname}