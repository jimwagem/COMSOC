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
    num_correct = 0
    num_total = 0
    for x, target in val_dataset:
        y = model(x)
        loss_list.append(criterion(y, target).item())
        correct = (y>0) == (x>0)
        num_correct += torch.sum(correct).item()
        num_total += len(x)
    print(f'average validation loss: {np.mean(loss_list)}, accuracy: {num_correct/num_total:.4f}')
    # TODO: Print election results

def train(model, train_dataset, epochs=5, batch_size=1):
    # Init dataset
    trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle = True)

    # Init Autoencoder
    criterion = nn.MSELoss()
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    losses = []
    for i in range(epochs):
        print('epoch', i)
        for j, (x, target) in enumerate(trainloader):
            if j % (2000//batch_size) == 0 and j > 0:
                print('TRAIN LOSS:', torch.mean(torch.Tensor(losses)).item())
                losses = []
            optimizer.zero_grad()

            # Forward
            y = model.forward(x)
            loss = criterion(y, target)
            # print(torch.sum(torch.max(torch.zeros(y.shape), y)).item(), torch.sum(torch.max(torch.zeros(y.shape), target)).item())
            losses.append(loss.item())

            # backward
            loss.backward()
            optimizer.step()

    return model

if __name__ == '__main__':
    dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = 0.25)
    num_ballots = len(dataset)
    num_val = num_ballots//2
    num_train = num_ballots - num_val
    train_dataset, val_dataset = data.random_split(dataset, [num_train ,num_val])

    model = AutoEncoder(dataset.n_projects, [], dataset.n_projects)
    train(model, train_dataset, epochs=20, batch_size=10)
    validate(model, val_dataset)
    # completed = model.complete_ballots(dataset.x)
    # print(completed)
