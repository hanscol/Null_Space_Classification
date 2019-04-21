from __future__ import print_function, division
import torch
import numpy as np


def train(model, device, loader, optimizer, config):
    model.train()

    correct = 0
    total_loss = 0

    for batch_idx, sample in enumerate(loader):

        # if (batch_idx+1) * loader.batch_size < loader.dataset.stand_len:
        data = sample['image']
        target = sample['target']

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_fun = torch.nn.CrossEntropyLoss()

        if config.null_space_tuning:
            null_data1 = sample['null_img1']
            null_data2 = sample['null_img2']

            null_data1, null_data2 = null_data1.to(device), null_data2.to(device)

            null_out1 = model(null_data1)
            null_out2 = model(null_data2)

            null_loss_fun = torch.nn.MSELoss()
            # if (batch_idx + 1) * loader.batch_size < loader.dataset.stand_len:
            loss = loss_fun(output, target) + config.alpha*null_loss_fun(null_out1, null_out2)
            # else:
            #     loss = (config.alpha/10)*null_loss_fun(null_out1, null_out2)
        else:
            loss = loss_fun(output, target)

        # if (batch_idx + 1) * loader.batch_size < loader.dataset.stand_len:
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view(-1, 1)).sum().item()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / (batch_idx+1)
    accuracy = 100*(correct / loader.dataset.len)

    print('\tTraining set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(avg_loss, accuracy))

    return total_loss, accuracy


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