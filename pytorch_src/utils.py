from __future__ import print_function, division
import torch
import numpy as np
from vat import VATLoss


def train(model, device, loader, optimizer, config):
    model.train()

    correct = 0
    total_loss = 0

    for batch_idx, sample in enumerate(loader):
        data = sample['image']
        target = sample['target']

        if config.vat:
            vat_loss = VATLoss(xi=config.xi, eps=config.eps, ip=config.ip)
            ul_data = sample['ul_image']
            ul_data = ul_data.to(device)
            lds = vat_loss(model, ul_data)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_fun = torch.nn.CrossEntropyLoss()

        if config.null_space_tuning:
            ul_data1 = sample['ul_img1']
            ul_data2 = sample['ul_img2']

            ul_data1, ul_data2 = ul_data1.to(device), ul_data2.to(device)

            ul_out1 = model(ul_data1)
            ul_out2 = model(ul_data2)

            null_loss_fun = torch.nn.MSELoss()
            loss = loss_fun(output, target) + config.alpha*null_loss_fun(ul_out1, ul_out2)
        elif config.vat:
            loss = loss_fun(output, target) + config.alpha*lds
        else:
            loss = loss_fun(output, target)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view(-1, 1)).sum().item()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / (batch_idx+1)
    accuracy = 100*(correct / loader.dataset.len)

    print('\tTraining set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(avg_loss, accuracy))

    return avg_loss, accuracy


def test(model, device, loader):
    model.eval()

    correct = 0
    total_loss = 0
    correct_data = []
    incorrect_data = []

    init = True
    confusion = None

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            data = sample['image']
            target = sample['target']
            file = sample['file']

            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.to(device)

            if init:
                confusion = np.zeros((output.shape[1],output.shape[1]))
                init = False
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(output, target)

            pred = output.max(1, keepdim=True)[1]
            correct_mask = pred.eq(target.view(-1,1))
            correct += correct_mask.sum().item()

            for i in range(len(correct_mask)):
                if correct_mask[i] == 1:
                    correct_data.append([file[i], int(pred[i]), int(target[i])])
                else:
                    incorrect_data.append([file[i], int(pred[i]), int(target[i])])

                confusion[int(pred[i]), int(target[i])] += 1

            total_loss += loss.item()

    avg_loss = total_loss / (batch_idx+1)
    accuracy = 100 * (correct / len(loader.dataset))

    print('\tTesting set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(avg_loss, accuracy))

    return total_loss, accuracy, confusion, correct_data, incorrect_data
