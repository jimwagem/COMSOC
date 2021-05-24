import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time

from synthdata import SynthDataLoader
from autoencoder import AutoEncoder
from dataloader import RealDataLoader
from election import ballot_distance, greedy_approval

def election_accuracy(true_ballots, completed_ballots, budget, project_costs):
    true_ap = greedy_approval(true_ballots, budget, project_costs)
    completed_ap = greedy_approval(completed_ballots, budget, project_costs)
    num_correct_projects=0
    true_used_budget = 0
    completed_correct_budget = 0
    for t_project in true_ap:
        true_used_budget += project_costs[t_project]
        for c_project in completed_ap:
            if c_project == t_project:
                num_correct_projects += 1
                completed_correct_budget += project_costs[c_project]
    print(f'Project accuracy {num_correct_projects/len(true_ap):.4f}')
    print(f'Budget accuracy {completed_correct_budget/true_ballots:.4f}')


def validate(model, val_dataset, device='cpu'):
    criterion = nn.MSELoss()
    loss_list = []
    # accuracy of total reconstruction
    num_correct = 0
    num_total = 0
    # accuracy of missing reconstruction
    num_filled_correct = 0
    num_filled_total = 0
    # accuracy of missing reconstruction if we always filled in -1 or 1
    num_neg_ones_correct = 0
    model.eval()
    for x, target in val_dataset:
        x = x.to(device)
        target = target.to(device)
        y = model(x.unsqueeze(dim=0)).squeeze()

        loss_list.append(criterion(y, target).item())
        correct = (y>0) == (target>0)
        correct = correct.to('cpu')
        num_correct += torch.sum(correct).item()
        num_total += len(x)

        zeros = torch.zeros(target.shape)
        ones = torch.ones(target.shape)

        filled = torch.where(x.to('cpu') == 0, ones, zeros)
        filled_correct = filled*correct
        num_filled_total += torch.sum(filled)
        num_filled_correct += torch.sum(filled_correct)

        num_neg_ones_correct += torch.sum((target.to('cpu') == -1)*filled)

    print(f'average validation loss: {np.mean(loss_list)}, accuracy: {num_correct/num_total:.4f}')
    print(f'accuracy on filled: {num_filled_correct/num_filled_total:.4f}')
    print(f'accuracy if only -1: {num_neg_ones_correct/num_filled_total:.4f}')
    print(f'accuracy if only 1: {1 - num_neg_ones_correct/num_filled_total:.4f}')
    # TODO: Print election results

def train(model, dataset, epochs=5, batch_size=1, device='cpu', use_project_costs=True):
    # Init dataset
    trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle = True)
    model.to(device)

    criterion = nn.MSELoss()

    if use_project_costs:
        pc = dataset.project_costs
    else:
        pc = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    losses = []
    for i in range(epochs):
        print('epoch', i)
        for j, (x, target) in enumerate(trainloader):
            x = x.to(device)
            target = target.to(device)
            if j % (2000//batch_size) == 0 and j > 0:
                print('TRAIN LOSS:', torch.mean(torch.Tensor(losses)).item())
                losses = []
            optimizer.zero_grad()

            # Forward
            y = model.forward(x)
            # loss = criterion(y, target)
            mask = 1 - torch.abs(x)
            # mask=None
            loss = ballot_distance(y, target, L1=True, mask=mask)
            # print(torch.sum(torch.max(torch.zeros(y.shape), y)).item(), torch.sum(torch.max(torch.zeros(y.shape), target)).item())
            losses.append(loss.item())

            # backward
            loss.backward()
            optimizer.step()
    print(torch.mean(mask))
    return model

if __name__ == '__main__':
    start=time.time()
    device='cuda'
    # dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = 0.25, mask_per_ballot=5)
    dataset = SynthDataLoader(11,10000,100)
    num_ballots = len(dataset)
    num_val = num_ballots//3
    num_train = num_ballots - num_val
    train_dataset, val_dataset = data.random_split(dataset, [num_train,num_val])

    model = AutoEncoder(dataset.n_projects, [70,50,40], 30)
    train(model, train_dataset, epochs=250, batch_size=32, device=device)
    validate(model, val_dataset, device)
    end=time.time()
    duration = int(end-start)
    print(f'took {duration//60} minutes and {duration%60} seconds')
    # completed = model.complete_ballots(dataset.x)
    # print(completed)
