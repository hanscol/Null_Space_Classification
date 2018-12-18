from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from torchvision import models
from data import *
from model import AllConvNet
from utils import *
import random

def train_model(null_split):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_data = train_files('/home/local/VANDERBILT/hansencb/CIFAR-10/Train')
    test_data = train_files('/home/local/VANDERBILT/hansencb/CIFAR-10/Validate')

    train_dataset = Train_Dataset(train_data, null_split=null_split)
    test_dataset = Train_Dataset(test_data)


    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    #model = AllConvNet(3).to(device)

    model = models.densenet121(pretrained=True)
    ft = model.classifier.in_features
    model.classifier = torch.nn.Linear(ft, 10)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_loss_file = 'results/train_loss_split_{}.txt'.format(null_split)
    f = open(train_loss_file, 'w')
    f.close()
    validate_loss_file = 'results/validate_loss_split_{}.txt'.format(null_split)
    f = open(validate_loss_file, 'w')
    f.close()
    train_accuracy_file = 'results/train_accuracy_split_{}.txt'.format(null_split)
    f = open(train_accuracy_file, 'w')
    f.close()
    validate_accuracy_file = 'results/validate_accuracy_split_{}.txt'.format(null_split)
    f = open(validate_accuracy_file, 'w')
    f.close()

    model_file = 'models/saved_model_split_{}'.format(null_split)
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

        loss, accuracy = test(model, device, test_loader)

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

        if loss > last_loss:
            seq_increase += 1
            if seq_increase > 2:
                break
        else:
            seq_increase = 0

        last_loss = loss

def main():
    #splits = [1000, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 44000]
    #splits = [1000, 4000, 8000, 12000]
    #splits = [16000, 20000, 24000, 28000]
    splits = [44000]
    splits.reverse()
    for split in splits:
        print('Training model with split {}'.format(split))
        train_model(split)

if __name__ == '__main__':
    main()
