import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time

from autoencoder import AutoEncoder
from dataloader import RealDataLoader

def validate(model, val_dataset):
    criterion = nn.MSELoss()
    loss_list = []
    # accuracy of total reconstruction
    num_correct = 0
    num_total = 0
    # accuracy of missing reconstruction
    num_filled_correct = 0
    num_filled_total = 0
    # accuracy of missing reconstruction if we always filled in -1
    num_neg_ones_correct = 0

    for x, target in val_dataset:
        y = model(x)

        loss_list.append(criterion(y, target).item())
        correct = (y>0) == (target>0)
        num_correct += torch.sum(correct).item()
        num_total += len(target)

        zeros = torch.zeros(target.shape)
        ones = torch.ones(target.shape)

        filled = torch.where(x == 0, ones, zeros)
        filled_correct = filled*correct
        num_filled_total += torch.sum(filled)
        num_filled_correct += torch.sum(filled_correct)

        num_neg_ones_correct += torch.sum((target == -1)*filled)

    print(f'average validation loss: {np.mean(loss_list)}, accuracy: {num_correct/num_total:.4f}')
    print(f'accuracy on filled: {num_filled_correct/num_filled_total:.4f}')
    print(f'accuracy if only -1: {num_neg_ones_correct/num_filled_total:.4f}')
    # TODO: Print election results

def train(model, train_dataset, epochs=5, batch_size=1):
    # Init dataset
    trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle = True)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    losses = []
    for i in range(epochs):
        print('epoch', i, end = ' ')
        for j, (x, target) in enumerate(trainloader):

            optimizer.zero_grad()

            # Forward
            y = model.forward(x)
            zeros = torch.zeros(target.shape)
            ones = torch.ones(target.shape)
            filled = torch.where(x == 0, ones, zeros)
            loss = criterion(y*filled, target*filled)
            # print(torch.sum(torch.max(torch.zeros(y.shape), y)).item(), torch.sum(torch.max(torch.zeros(y.shape), target)).item())
            losses.append(loss.item())

            # backward
            loss.backward()
            optimizer.step()
        print('TRAIN LOSS:', torch.mean(torch.Tensor(losses)).item())
        losses = []

    return model

if __name__ == '__main__':
    dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = 0.1)
    num_ballots = len(dataset)
    num_val = num_ballots//3
    num_train = num_ballots - num_val
    train_dataset, val_dataset = data.random_split(dataset, [num_train ,num_val])

    model = AutoEncoder(dataset.n_projects, [75], 50)
    train(model, train_dataset, epochs=25, batch_size=10)
    validate(model, val_dataset)
    # completed = model.complete_ballots(dataset.x)
    # print(completed)
