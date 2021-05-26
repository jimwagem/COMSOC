import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time
import argparse

from autoencoder import AutoEncoder
from dataloader import RealDataLoader
from synthdata import SynthDataLoader
from election import ballot_distance
from evaluation import evaluate_acc, evaluate_outcome

def train(model, train_dataset, epochs=5, batch_size=1, verbose = True, pc = None, device='cpu', use_mask=True):
    # Init dataset
    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    criterion = nn.MSELoss()
    model.to(device)

    if pc is not None:
        pc = pc.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    losses = []
    for i in range(epochs):
        if verbose:
            print('epoch', i, end = ' ')
        for j, (x, target) in enumerate(trainloader):
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            # Forward
            y = model.forward(x)

            if use_mask:
                mask = 1 - torch.abs(x)
            else:
                mask= torch.ones_like(x)
            loss = criterion(y*mask, target*mask)
            # loss = ballot_distance(y, target, L1=True, mask=mask, project_costs=pc)
            # print(torch.sum(torch.max(torch.zeros(y.shape), y)).item(), torch.sum(torch.max(torch.zeros(y.shape), target)).item())

            # backward
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        # scheduler.step()
        if verbose:
            print('TRAIN LOSS:', torch.mean(torch.Tensor(losses)).item())
        losses = []
    model.cpu()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ballot completion using an AE.')
    parser.add_argument('--dropout', '-d', type=float, default=0.25,
                    help='How much of the data should be dropped out.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs to train for.')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    start = time.time()

    # dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = args.dropout)
    dataset = SynthDataLoader(num_categories=5, num_voters=8000, num_projects=80, dropout=0.25)
    num_ballots = len(dataset)
    num_val = num_ballots//4
    num_train = num_ballots - num_val
    train_dataset, val_dataset = data.random_split(dataset, [num_train ,num_val])

    model = AutoEncoder(dataset.n_projects, [54], 50)
    # pc = dataset.project_costs
    pc = None
    train(model, train_dataset, epochs=args.epochs, batch_size=32, pc = pc, device=args.device, verbose=True, use_mask=True)

    evaluate_acc(model, val_dataset)
    evaluate_outcome(model, dataset, val_dataset)
    end = time.time()
    duration = end-start
    print(f'Took {duration//60} minutes and {duration%60} seconds')
