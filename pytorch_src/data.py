from __future__ import print_function, division
import torch

from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import random
from torchvision import transforms
from PIL import Image
from tqdm import tqdm, trange
import warnings

warnings.filterwarnings("ignore")


class Train_Dataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.keys = list(data.keys())
        self.loaded_imgs = {}
        self.config = config
        self.unlabelled_split = int(config.unlabelled_split * len(self.keys))
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224,224))
        self.transforms = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.1, 0.1), shear=2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter()
        ])

        self.load_imgs(self.keys)

        self.class_to_files = {}
        for key in self.keys:
            label = self.data[key]
            if label not in self.class_to_files:
                self.class_to_files[label] = [key]
            else:
                self.class_to_files[label].append(key)

        class_splits = []
        for key in range(len(self.class_to_files)):
            class_splits.append(int(self.unlabelled_split * (len(self.class_to_files[key]) / len(self.keys))))

        self.ul_class_to_files = {}
        if self.unlabelled_split > 0:
            diff = self.unlabelled_split - sum(class_splits)
            class_splits[np.argmax(class_splits)] += diff
            for i in range(len(class_splits)):
                class_split = class_splits[i]
                keys = self.class_to_files[i]
                split_keys = keys[0:class_split]

                self.ul_class_to_files[i] = split_keys
                self.class_to_files[i] = keys[class_split:]


        self.len = 0
        if config.dataset == 'HAM10000':
            for i in range(len(self.class_to_files)):
                self.len = max(self.len, len(self.class_to_files[i]))
            self.len *= i
        else:
            for i in range(len(self.class_to_files)):
                self.len += len(self.class_to_files[i])

        self.old_class_to_files = self.class_to_files.copy()

    def load_imgs(self, fnames):
        print('Loading images:')
        for fname in tqdm(fnames):
            img = io.imread(fname)

            if self.config.dataset == 'HAM10000':
                img = Image.fromarray(img.astype('uint8'))
                img = self.resize(img)

            self.loaded_imgs[fname] = img

    def bootstrap(self, model, device):
        self.class_to_files = self.old_class_to_files.copy()
        model.eval()
        with torch.no_grad():
            for c in self.ul_class_to_files:
                for fname in self.ul_class_to_files[c]:
                    img = self.preprocess(fname).unsqueeze(0)
                    img = img.to(device)
                    output = model(img)
                    label = torch.argmax(output)
                    self.class_to_files[label.item()].append(fname)

    def __len__(self):
        return self.len

    def preprocess(self, fname):
        img = self.loaded_imgs[fname]

        if self.config.dataset == 'MNIST':
            img = img / 255.0

            img = np.expand_dims(img, axis=2)
            img = self.tensor(img)
            img = img.type(torch.float32)

        if self.config.dataset == 'CIFAR10':
            img = Image.fromarray(img.astype('uint8'))
            img = self.resize(img)
            img = self.tensor(img)
            img = self.norm(img.float())
            img = img.type(torch.float32)

        if self.config.dataset == 'HAM10000':
            if self.transforms:
                img = self.transforms(img)

            img = self.tensor(img)
            img = self.norm(img.float())
            img = img.type(torch.float32)

        return img

    def __getitem__(self, i):
        label = random.randint(0, len(self.class_to_files) - 1)
        idx = random.randint(0, len(self.class_to_files[label]) - 1)
        fname = self.class_to_files[label][idx]

        self.idx = i
        img = self.preprocess(fname)

        if self.config.null_space_tuning:
            ul_label = random.randint(0, len(self.class_to_files) - 1)
            ul_keys = self.ul_class_to_files[ul_label].copy()
            idx = random.randint(0, len(ul_keys) - 1)
            ul_image1 = self.preprocess(ul_keys[idx])
            del ul_keys[idx]
            idx = random.randint(0, len(ul_keys) - 1)
            ul_image2 = self.preprocess(ul_keys[idx])

            return {'image': img, \
                    'target': torch.tensor(label), \
                    'ul_img1': ul_image1, \
                    'ul_img2': ul_image2}
        elif self.config.vat:
            ul_label = random.randint(0, len(self.class_to_files) - 1)
            ul_keys = self.ul_class_to_files[ul_label].copy()
            idx = random.randint(0, len(ul_keys) - 1)
            ul_image1 = self.preprocess(ul_keys[idx])

            return {'image': img, \
                    'target': torch.tensor(label), \
                    'ul_image': ul_image1}
        else:
            return {'image': img, \
                    'target': torch.tensor(label)}


class Test_Dataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.keys = list(data.keys())
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224))
        self.config = config
        self.loaded_imgs = {}

        self.load_imgs(self.keys)

    def __len__(self):
        return len(self.keys)

    def load_imgs(self, fnames):
        print('Loading images:')
        for fname in tqdm(fnames):
            img = io.imread(fname)

            if self.config.dataset == 'HAM10000':
                img = Image.fromarray(img.astype('uint8'))
                img = self.resize(img)

            self.loaded_imgs[fname] = img

    def preprocess(self, fname):
        img = self.loaded_imgs[fname]

        if self.config.dataset == 'MNIST':
            img = img / 255.0
            img = np.expand_dims(img, axis=2)
            img = self.tensor(img)

        if self.config.dataset == 'CIFAR10':
            img = Image.fromarray(img.astype('uint8'))
            img = self.resize(img)
            img = self.tensor(img)
            img = self.norm(img.float())

        if self.config.dataset == 'HAM10000':
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

    def set_use_cache(self):
        self.loaded_imgs = torch.stack(self.loaded_imgs)

