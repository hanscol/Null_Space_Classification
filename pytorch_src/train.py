from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from torchvision import models
from data import *
from utils import *
import random
import argparse
import os

def train_model(config):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data = {}
    with open(config.file, 'r') as f:
        for l in f.readlines():
            l = l.split(' ')
            data[l[0]] = int(l[1])


    keys = list(data.keys())
    val_split = (len(keys)*(1-config.null_split))*config.val_split
    val_data = {}
    for i in range(val_split):
        idx = random.randint(0,len(keys)-1)
        val_data[keys[idx]] = data[keys[idx]]
        del data[keys[idx]]

    train_dataset = Train_Dataset(data, config)
    test_dataset = Test_Dataset(val_data, config)


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)


    if config.dataset == 'MNIST':
        model = CNN(1, 10)
    if config.dataset == 'CIFAR-10':
        model = models.densenet121(pretrained=True)
        ft = model.classifier.in_features
        model.classifier = torch.nn.Linear(ft, 10)
    if config.dataset == 'HAM10000':
        model = models.densenet169(pretrained=True)
        ft = model.classifier.in_features
        model.classifier = torch.nn.Linear(ft, 7)

    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loss_file = '{}/results/train_loss_split_{}.txt'.format(config.out_dir, config.null_split)
    f = open(train_loss_file, 'w')
    f.close()
    validate_loss_file = '{}/results/validate_loss_split_{}.txt'.format(config.out_dir, config.null_split)
    f = open(validate_loss_file, 'w')
    f.close()
    train_accuracy_file = '{}/results/train_accuracy_split_{}.txt'.format(config.out_dir, config.null_split)
    f = open(train_accuracy_file, 'w')
    f.close()
    validate_accuracy_file = '{}/results/validate_accuracy_split_{}.txt'.format(config.out_dir, config.null_split)
    f = open(validate_accuracy_file, 'w')
    f.close()

    model_file = '{}/models/saved_model_split_{}'.format(config.out_dir, config.null_split)
    seq_increase = 0
    min_loss = 10000
    last_loss = 10000

    for epoch in range(1, 11):
        print('\nEpoch %d: ' % epoch)
        loss, accuracy = train(model, device, train_loader, optimizer)

        with open(train_loss_file, "a") as file:
            file.write(str(loss))
            file.write('\n')
        with open(train_accuracy_file, "a") as file:
            file.write(str(accuracy))
            file.write('\n')

        loss, accuracy, confusion, correct_data, incorrect_data = test(model, device, test_loader)

        with open(validate_loss_file, "a") as file:
            file.write(str(loss))
            file.write('\n')
        with open(validate_accuracy_file, "a") as file:
            file.write(str(accuracy))
            file.write('\n')

        if loss < min_loss:
            min_loss = loss
            with open(model_file, 'wb') as f:
                torch.save(model.state_dict(), f)

        if config.early_stop != -1:
            if loss > last_loss:
                seq_increase += 1
                if seq_increase == config.early_stop:
                    break
            else:
                seq_increase = 0
            last_loss = loss

        if config.decay_epoch != -1:
            if epoch % config.decay_epoch == 0:
                config.lr = config.lr * config.decay_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.lr

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='MNIST', help='Three datasets may be chosen from (MNIST, CIFAR-10, HAM1000)')
    parser.add_argument('--null_space_tuning', type=bool, default=False, help='Determines if a standard network or a null space tuning network will be used')
    parser.add_argument('--null_split', type=float, default=0.10, help='Determines the amount of the training data will have the labels withheld')
    parser.add_argument('--val_split', type=float, default=0.05, help='Determines the amount of the training data will be used for validation during training (after null_split)')
    parser.add_argument('--file', type=str, default='train.txt', help='This file should contain the path to an image as well and an integer specifying its class (space separated) on each line')
    parser.add_argument('--out_dir', type=str, default='out/', help='Path to output directory')
    parser.add_argument('--mode', type=str, default='train', help='Determines whether to backpropagate or not')
    parser.add_argument('--batch_size', type=int, default=16, help='Decides size of each training batch')
    parser.add_argument('--lr', type=float, default=0.0001, help='Optimizer\'s learning rate')
    parser.add_argument('--total_epochs', type=int, default=10, help='Maximum number of epochs for training')
    parser.add_argument('--early_stop', type=int, default=-1, help='Decide to stop early after this many epochs in which the validation loss increases (-1 means no early stopping)')
    parser.add_argument('--decay_epoch', type=int, default=-1, help='Decay the learning rate after every this many epochs (-1 means no lr decay)')
    parser.add_argument('--decay_rate', type=float, default=0.10, help='Rate at which the learning rate will be decayed')

    config = parser.parse_args()

    if config.mode == 'train':
        train_model(config)
    elif config.mode == 'test':

if __name__ == '__main__':
    main()
