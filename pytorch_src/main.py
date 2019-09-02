from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from torchvision import models
from pytorch_src.data import *
from pytorch_src.utils import *
from pytorch_src.model import *
import random
import argparse
import os

def test_model(config):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data = {}
    with open(config.file, 'r') as f:
        for l in f.readlines():
            l = l.split(' ')
            data[l[0]] = int(l[1])

    test_dataset = Test_Dataset(data, config)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

    if config.dataset == 'MNIST':
        model = CNN(1, 10)
    if config.dataset == 'CIFAR10':
        model = models.densenet121(pretrained=True)
        ft = model.classifier.in_features
        model.classifier = torch.nn.Linear(ft, 10)
    if config.dataset == 'HAM10000':
        model = models.densenet169(pretrained=True)
        ft = model.classifier.in_features
        model.classifier = torch.nn.Linear(ft, 7)

    model_file = '{}/models/saved_model_split_{}'.format(config.out_dir, config.unlabelled_split)
    model.load_state_dict(torch.load(model_file))
    model = model.to(device)

    loss, accuracy, conf_matrix, correct, incorrect = test(model, device, test_loader)

    correct_file = '{}/correct_lists/list_correct_model_split_{}'.format(config.out_dir, config.unlabelled_split)
    with open(correct_file, 'w') as f:
        for i in correct:
            line = '{} {} {}\n'.format(i[0], str(i[1]), str(i[2]))
            f.write(line)
        for i in incorrect:
            line = '{} {} {}\n'.format(i[0], str(i[1]), str(i[2]))
            f.write(line)

<<<<<<< HEAD
    acc_file = '{}/results/test_accuracy_split_{}'.format(config.out_dir, config.unlabelled_split)
=======
    acc_file = '{}/results/test_accuracy_split_{}'.format(config.out_dir, config.null_split)
>>>>>>> 735066cd8f3de8328dde38f34c42daf21c6dcf86
    with open(acc_file, 'w') as f:
        f.write(str(accuracy))

    return accuracy

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
    val_split = int((len(keys)*(1-config.unlabelled_split))*config.val_split)
    val_data = {}
    for i in range(val_split):
        idx = random.randint(0,len(keys)-1)
        val_data[keys[idx]] = data[keys[idx]]
        del data[keys[idx]]
        del keys[idx]

    train_dataset = Train_Dataset(data, config)
<<<<<<< HEAD
    val_dataset = Test_Dataset(val_data, config)
=======
    #val_dataset = Test_Dataset(val_data, config)
    test_data = {}
    with open('../MNIST_Null_Space_Tuning/k_fold_files/test_fold_0.txt', 'r') as f:
        for l in f.readlines():
            l = l.split(' ')
            test_data[l[0]] = int(l[1])
    test_dataset = Test_Dataset(test_data, config)


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
    #val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
>>>>>>> 735066cd8f3de8328dde38f34c42daf21c6dcf86

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

    if config.dataset == 'MNIST':
        model = CNN(1, 10)
    if config.dataset == 'CIFAR10':
        model = models.densenet121(pretrained=True)
        ft = model.classifier.in_features
        model.classifier = torch.nn.Linear(ft, 10)
    if config.dataset == 'HAM10000':
        model = models.densenet169(pretrained=True)
        ft = model.classifier.in_features
        model.classifier = torch.nn.Linear(ft, 7)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loss_file = '{}/results/train_loss_split_{}.txt'.format(config.out_dir, config.unlabelled_split)
    f = open(train_loss_file, 'w')
    f.close()
    validate_loss_file = '{}/results/validate_loss_split_{}.txt'.format(config.out_dir, config.unlabelled_split)
    f = open(validate_loss_file, 'w')
    f.close()
    train_accuracy_file = '{}/results/train_accuracy_split_{}.txt'.format(config.out_dir, config.unlabelled_split)
    f = open(train_accuracy_file, 'w')
    f.close()
    validate_accuracy_file = '{}/results/validate_accuracy_split_{}.txt'.format(config.out_dir, config.unlabelled_split)
    f = open(validate_accuracy_file, 'w')
    f.close()

<<<<<<< HEAD
    model_file = '{}/models/saved_model_split_{}'.format(config.out_dir, config.unlabelled_split)
=======
    model_file = '{}/models/saved_model_split_{}'.format(config.out_dir, config.null_split)
>>>>>>> 735066cd8f3de8328dde38f34c42daf21c6dcf86

    if config.continue_training:
        model.load_state_dict(torch.load(model_file))
        model = model.to(device)

    seq_increase = 0
    min_loss = 10000
<<<<<<< HEAD

    bootstraps = 0
    if config.bootstrap:
        bootstraps = 10
    boot_init = True

=======
    last_loss = 10000

    bootstraps = 0
    if config.bootstrap:
        bootstraps = 1

    # alpha_increasing = True
>>>>>>> 735066cd8f3de8328dde38f34c42daf21c6dcf86
    for bootstrap in range(bootstraps+1):
        for epoch in range(1, config.total_epochs+1):
            print('\nEpoch %d: ' % epoch)
            loss, accuracy = train(model, device, train_loader, optimizer, config)

            with open(train_loss_file, "a") as file:
                file.write(str(loss))
                file.write('\n')
            with open(train_accuracy_file, "a") as file:
                file.write(str(accuracy))
                file.write('\n')

<<<<<<< HEAD
            loss, accuracy, confusion, correct_data, incorrect_data = test(model, device, val_loader)
=======
            #loss, accuracy, confusion, correct_data, incorrect_data = test(model, device, val_loader)
            loss, accuracy, confusion, correct_data, incorrect_data = test(model, device, test_loader)
>>>>>>> 735066cd8f3de8328dde38f34c42daf21c6dcf86

            class_accuracies = '\t\tAccuracy by class: '
            for i in range(len(confusion)):
                correct = confusion[i,i]
                total = sum(confusion[:,i])
                class_accuracies += '{0}--{1:.2f}%   '.format(i, (correct/total)*100)

            print(class_accuracies)


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

<<<<<<< HEAD
            elif config.early_stop != -1:
                if loss > min_loss:
=======
            if config.early_stop != -1:
                if loss > last_loss:
>>>>>>> 735066cd8f3de8328dde38f34c42daf21c6dcf86
                    seq_increase += 1
                    if seq_increase == config.early_stop:
                        break
                else:
                    seq_increase = 0
<<<<<<< HEAD
=======
                last_loss = loss
>>>>>>> 735066cd8f3de8328dde38f34c42daf21c6dcf86

            if config.decay_epoch != -1:
                if epoch % config.decay_epoch == 0:
                    config.lr = config.lr * config.decay_rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config.lr

        if config.bootstrap and bootstrap != bootstraps:
<<<<<<< HEAD
            if boot_init:
                parts = config.out_dir.split('/')
                parts[-1] = parts[-1] + '_bootstrap'
                config.out_dir = '/'.join(parts)

                if not os.path.isdir(config.out_dir):
                    os.mkdir(config.out_dir)
                if not os.path.isdir(os.path.join(config.out_dir, 'models')):
                    os.mkdir(os.path.join(config.out_dir, 'models'))
                if not os.path.isdir(os.path.join(config.out_dir, 'results')):
                    os.mkdir(os.path.join(config.out_dir, 'results'))
                if not os.path.isdir(os.path.join(config.out_dir, 'correct_lists')):
                    os.mkdir(os.path.join(config.out_dir, 'correct_lists'))

                train_loss_file = '{}/results/train_loss_split_{}.txt'.format(config.out_dir, config.unlabelled_split)
                f = open(train_loss_file, 'w')
                f.close()
                validate_loss_file = '{}/results/validate_loss_split_{}.txt'.format(config.out_dir, config.unlabelled_split)
                f = open(validate_loss_file, 'w')
                f.close()
                train_accuracy_file = '{}/results/train_accuracy_split_{}.txt'.format(config.out_dir, config.unlabelled_split)
                f = open(train_accuracy_file, 'w')
                f.close()
                validate_accuracy_file = '{}/results/validate_accuracy_split_{}.txt'.format(config.out_dir, config.unlabelled_split)
                f = open(validate_accuracy_file, 'w')
                f.close()

                model_file = '{}/models/saved_model_split_{}'.format(config.out_dir, config.unlabelled_split)
                boot_init = False
                config.total_epochs = int(config.total_epochs/2)
=======
            parts = config.out_dir.split('/')
            parts[-1] = parts[-1] + '_bootstrap'
            config.out_dir = '/'.join(parts)

            if not os.path.isdir(config.out_dir):
                os.mkdir(config.out_dir)
            if not os.path.isdir(os.path.join(config.out_dir, 'models')):
                os.mkdir(os.path.join(config.out_dir, 'models'))
            if not os.path.isdir(os.path.join(config.out_dir, 'results')):
                os.mkdir(os.path.join(config.out_dir, 'results'))
            if not os.path.isdir(os.path.join(config.out_dir, 'correct_lists')):
                os.mkdir(os.path.join(config.out_dir, 'correct_lists'))

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
>>>>>>> 735066cd8f3de8328dde38f34c42daf21c6dcf86
            train_dataset.bootstrap(model, device)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='MNIST', help='Three datasets may be chosen from (MNIST, CIFAR-10, HAM1000)')
<<<<<<< HEAD
    parser.add_argument('--null_space_tuning', action='store_true', help='Turn on null space training')
    parser.add_argument('--vat', action='store_true', help='Turn on virtual adversarial trianing.')
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI', help='hyperparameter of VAT (default: 10.0)')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',help='hyperparameter of VAT (default: 1.0)')
    parser.add_argument('--ip', type=int, default=1, metavar='IP', help='hyperparameter of VAT (default: 1)')
    parser.add_argument('--alpha', type=float, default=1, help='Determines the impact of null space tuning or VAT')
    parser.add_argument('--unlabelled_split', type=float, default=0.10, help='Determines the amount of the training data will have the labels withheld')
    parser.add_argument('--val_split', type=float, default=0.05, help='Determines the amount of the training data will be used for validation during training (after unlabelled_split)')
=======
    parser.add_argument('--null_space_tuning', action='store_true', help='Determines if a standard network or a null space tuning network will be used')
    parser.add_argument('--alpha', type=float, default=
    1, help='Determines the impact of null space tuning')
    parser.add_argument('--null_split', type=float, default=0.10, help='Determines the amount of the training data will have the labels withheld')
    parser.add_argument('--val_split', type=float, default=0.05, help='Determines the amount of the training data will be used for validation during training (after null_split)')
>>>>>>> 735066cd8f3de8328dde38f34c42daf21c6dcf86
    parser.add_argument('--file', type=str, default='train.txt', help='This file should contain the path to an image as well and an integer specifying its class (space separated) on each line')
    parser.add_argument('--out_dir', type=str, default='out/', help='Path to output directory')
    parser.add_argument('--mode', type=str, default='train', help='Determines whether to backpropagate or not')
    parser.add_argument('--batch_size', type=int, default=16, help='Decides size of each training batch')
    parser.add_argument('--lr', type=float, default=0.0001, help='Optimizer\'s learning rate')
    parser.add_argument('--total_epochs', type=int, default=10, help='Maximum number of epochs for training')
    parser.add_argument('--early_stop', type=int, default=-1, help='Decide to stop early after this many epochs in which the validation loss increases (-1 means no early stopping)')
    parser.add_argument('--decay_epoch', type=int, default=-1, help='Decay the learning rate after every this many epochs (-1 means no lr decay)')
    parser.add_argument('--decay_rate', type=float, default=0.10, help='Rate at which the learning rate will be decayed')
    parser.add_argument('--bootstrap', action='store_true', help='Use trained model to generate weak labels, and continue training')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from saved model')

    config = parser.parse_args()

    print(config)

    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)
    if not os.path.isdir(os.path.join(config.out_dir,'models')):
        os.mkdir(os.path.join(config.out_dir, 'models'))
    if not os.path.isdir(os.path.join(config.out_dir, 'results')):
        os.mkdir(os.path.join(config.out_dir, 'results'))
    if not os.path.isdir(os.path.join(config.out_dir, 'correct_lists')):
        os.mkdir(os.path.join(config.out_dir, 'correct_lists'))

    if config.mode == 'train':
        train_model(config)
    elif config.mode == 'test':
        test_model(config)

if __name__ == '__main__':
    main()
