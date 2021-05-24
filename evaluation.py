import types
from functools import partial

import torch
import torch.nn as nn
import numpy as np

from autoencoder import AutoEncoder
from election import greedy_approval


def val_dataset_incomplete(args):
    pass


def evaluate_acc(model, val_dataset, train_dataset, nn_fraction):
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
    val_dataset_incomplete = val_dataset.dataset.x_list
    train_dataset_incomplete = train_dataset.dataset.x_list
    total_incomplete = val_dataset_incomplete + train_dataset_incomplete
    top_k = round(nn_fraction * len(val_dataset_incomplete))

    real_ballots = []

    if isinstance(model, types.FunctionType):
        partial_model = partial(model, incomplete_ballots=val_dataset_incomplete, top_k=top_k)
    else:
        partial_model = model

    for x, target in val_dataset:
        real_ballots.append(target)

        y = partial_model(x)

        zeros = torch.zeros(target.shape)
        ones = torch.ones(target.shape)
        filled = torch.where(x == 0, ones, zeros)
        if isinstance(model, AutoEncoder):
            loss_list.append(criterion(y*filled, target*filled).item())
        correct = (y>0) == (target>0)
        num_correct += torch.sum(correct).item()
        num_total += len(target)

        filled_correct = filled*correct
        num_filled_total += torch.sum(filled)
        num_filled_correct += torch.sum(filled_correct)

        num_neg_ones_correct += torch.sum((target == -1)*filled)

    print(f'average validation loss: {np.mean(loss_list)}')
    print(f'accuracy: {num_correct/num_total:.4f}')
    print(f'accuracy on filled: {num_filled_correct/num_filled_total:.4f}')
    print(f'accuracy if only -1: {num_neg_ones_correct/num_filled_total:.4f}')

def evaluate_outcome(model, dataset, val_dataset):
    budget = dataset.budget
    project_costs = torch.Tensor([p.cost for p in dataset.projects])

    # Get full validation dataset
    full_x, full_target = zip(*[batch for batch in val_dataset])
    full_x = torch.stack(full_x)
    full_target = torch.stack(full_target)
    # Predict targets
    full_y = model.complete_ballots(full_x)

    target_set = set([i.item() for i in greedy_approval(full_target, budget, project_costs)])
    y_set = set([i.item() for i in greedy_approval(full_y, budget, project_costs)])
    u = target_set.union(y_set)
    i = target_set.intersection(y_set)
    d = y_set.difference(i)

    target_cost = sum([dataset.projects[id].cost for id in target_set])
    i_cost = sum([dataset.projects[id].cost for id in i])

    print(f'Allocated same: {i_cost}/{target_cost}, ({i_cost/target_cost:.3f})')
