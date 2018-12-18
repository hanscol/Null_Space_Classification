from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from data import *
from model import CNN
from utils import *
import random

def test_model(null_split):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_data = train_files('/home/hansencb/MNIST/Test')

    test_dataset = Test_Dataset(test_data)

    batch_size = 128
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    model = CNN(1,10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model_file = 'models/saved_model_split_{}'.format(null_split)
    model.load_state_dict(torch.load(model_file))

    loss, accuracy, conf_matrix, correct, incorrect = test(model, device, test_loader)

    correct_file = 'correct_lists/list_correct_model_split_{}'.format(null_split)
    with open(correct_file, 'w') as f:
        for i in correct:
            line = '{} {} {}\n'.format(i[0], str(i[1]), str(i[2]))
            f.write(line)
        for i in incorrect:
            line = '{} {} {}\n'.format(i[0], str(i[1]), str(i[2]))
            f.write(line)

    return accuracy


def main():
    splits = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 54000]

    test_acc_file = 'results/test_accuracy.txt'
    f = open(test_acc_file, 'w')
		
    for split in splits:
        print('Testing model with split {}'.format(split))
        accuracy = test_model(split)
        f.write(str(accuracy)+'\n')
    f.close()
		

if __name__ == '__main__':
    main()
