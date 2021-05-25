import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time
import argparse

from autoencoder import AutoEncoder
from dataloader import RealDataLoader
from election import ballot_distance
from evaluation import evaluate_acc, evaluate_outcome

def train(model, train_dataset, epochs=5, batch_size=1, use_project_costs=True, verbose = True, pc = None):
    # Init dataset
    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    losses = []
    for i in range(epochs):
        if verbose:
            print('epoch', i, end = ' ')
        for j, (x, target) in enumerate(trainloader):
            optimizer.zero_grad()
            # Forward
            y = model.forward(x)

            mask = 1 - torch.abs(x)
            #loss = criterion(y*mask, target*mask)
            loss = ballot_distance(y, target, L1=False, mask=mask, project_costs=None)
            # print(torch.sum(torch.max(torch.zeros(y.shape), y)).item(), torch.sum(torch.max(torch.zeros(y.shape), target)).item())
            losses.append(loss.item())

            # backward
            loss.backward()
            optimizer.step()

        if verbose:
            print('TRAIN LOSS:', torch.mean(torch.Tensor(losses)).item())
        losses = []

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ballot completion using an AE.')
    parser.add_argument('--dropout', '-d', type=float, default=0.25,
                    help='How much of the data should be dropped out.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs to train for.')
    args = parser.parse_args()


    dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = args.dropout)
    num_ballots = len(dataset)
    num_val = num_ballots//4
    num_train = num_ballots - num_val
    train_dataset, val_dataset = data.random_split(dataset, [num_train ,num_val])

    model = AutoEncoder(dataset.n_projects, [75], 50)
    train(model, train_dataset, epochs=args.epochs, batch_size=8, pc = dataset.project_costs)

    evaluate_acc(model, val_dataset)
    evaluate_outcome(model, dataset, val_dataset)
