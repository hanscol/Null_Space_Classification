from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from torchvision import models
from data import *
from utils import *
import random

def test_model(null_split):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_data = train_files('/home/hansencb/CIFAR-10/Test')

    test_dataset = Test_Dataset(test_data)

    batch_size = 8
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    model = models.densenet121(pretrained=True)
    ft = model.classifier.in_features
    model.classifier = torch.nn.Linear(ft, 10)
    model = model.to(device)

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
    splits = [1000, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 44000]

    test_acc_file = 'results/test_accuracy.txt'
    f = open(test_acc_file, 'w')
		
    for split in splits:
        print('Testing model with split {}'.format(split))
        accuracy = test_model(split)
        f.write(str(accuracy)+'\n')
    f.close()
		

if __name__ == '__main__':
    main()
