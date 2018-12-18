from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from data import *
from model import CNN
from utils import *
import random

def train_model(null_split):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_data = train_files('/home/local/VANDERBILT/hansencb/MNIST/Train')
    test_data = train_files('/home/local/VANDERBILT/hansencb/MNIST/Validate')

    train_dataset = Train_Dataset(train_data, null_split=null_split)
    test_dataset = Train_Dataset(test_data)


    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    model = CNN(1,10).to(device)
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

    for epoch in range(1, 101):
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

        if epoch % 5 == 0:
            with open(model_file, 'wb') as f:
                torch.save(model.state_dict(), f)

def main():
    #splits = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 54000]
	splits = [50000, 54000]
	for split in splits:
		train_model(split)

if __name__ == '__main__':
    main()
