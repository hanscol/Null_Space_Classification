from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from torchvision import models
from data import *
from model import *
from utils import *


def test_model(null_split):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_data = train_files('/home/hansencb/SkinLesions/Test')

    test_dataset = Test_Dataset(test_data)

    batch_size = 16
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, **kwargs)

    model = models.densenet169(pretrained=True)
    ft = model.classifier.in_features
    model.classifier = torch.nn.Linear(ft, 7)

    model_file = 'models_bak/saved_model_split_{}'.format(null_split)
    model.load_state_dict(torch.load(model_file))
    model = model.to(device)

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
    splits = [1000, 2000, 3000, 4000, 5000, 5500, 6000, 6500, 7000, 7500, 7750, 8000]
    # splits = [1000, 2000, 3000]
    # test_acc_file = 'results/test_accuracy.txt'
    # f = open(test_acc_file, 'w')

    for split in splits:
        print('Testing model with split {}'.format(split))
        accuracy = test_model(split)
        print(accuracy)
        # f.write(str(accuracy)+'\n')
    # f.close()


if __name__ == '__main__':
    main()
