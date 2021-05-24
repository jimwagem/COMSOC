import collections
import random

import numpy as np
import torch
from torch.utils import data

from dataloader import RealDataLoader


class MissingPreferenceError(Exception):
    def __init__(self):
        super().__init__("No ballot contains a preference on this project")


def intersection_score(ballot, candidate):
    """determines the number of projects that two ballots both approve of"""
    return torch.sum(torch.where((ballot == 1) & (candidate == 1), 1, 0)).item()


def get_sorted_candidates(ballot, candidates):
    """computes the intersection scores and returns a sorted candidate-score dictionary"""
    scores = {}
    for candidate in candidates:
        scores[candidate] = intersection_score(ballot, candidate)
    # sort the score dictionary by intersection score value
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores


def get_majority_outcome(nearest_neighbors, index):
    """finds a nearest neighbor ballot for a project corresponding to index"""
    majority_score = 0
    # most_common = collections.Counter([x[0][index].item() for x in nearest_neighbors]).most_common(1)[0][0]
    for candidate in nearest_neighbors:
        majority_score += candidate[0][index]
    outcome = np.sign(majority_score) if majority_score != 0 else random.choice((-1, 1))
    # assert most_common == outcome
    return outcome


def max_intersection_completion(incomplete_ballots, top_k):
    """completes a set of ballots using nearest neighbor with respect to the intersection score"""
    completed_ballots = []
    for ballot in incomplete_ballots:
        # complete the ballot using the nearest neighbor's preference
        completed_ballot = single_mi_completion(ballot, incomplete_ballots, top_k)
        completed_ballots.append(completed_ballot)
    return completed_ballots


def single_mi_completion(ballot, incomplete_ballots, top_k):
    """completes a single ballot using a project-wise nearest neighbour approach based on the intersection score"""
    # identify the projects for which there is an incomplete preference
    missing_indices = torch.nonzero(torch.where(ballot == 0, 1, 0), as_tuple=True)[0].tolist()
    # possible nearest neighbors should exclude the ballot under consideration
    nn_candidates = [x for x in incomplete_ballots if not torch.equal(x, ballot)]
    # compute max intersection scores and sort accordingly
    nearest_neighbors = get_sorted_candidates(ballot, nn_candidates)[:top_k]
    completed_ballot = ballot.clone()
    for index in missing_indices:
        # for each missing project, find the nearest neighbor who has a completed preference on it
        outcome = get_majority_outcome(nearest_neighbors, index)
        completed_ballot[index] = outcome
    return completed_ballot


def validate_baseline(val_dataset, top_k):
    # accuracy of total reconstruction
    num_correct = 0
    num_total = 0
    # accuracy of missing reconstruction
    num_filled_correct = 0
    num_filled_total = 0
    # accuracy of missing reconstruction if we always filled in -1
    num_neg_ones_correct = 0
    num_ones_correct = 0
    val_dataset_incomplete = val_dataset.dataset.x_list
    val_dataset_complete = val_dataset.dataset.targets_list
    prop_neg_one = sum([torch.sum(torch.where(x == -1, 1, 0)) for x in val_dataset_complete]) / (
                len(val_dataset_complete) * len(val_dataset_complete[0]))
    for x, target in val_dataset:
        y = single_mi_completion(x, val_dataset_incomplete, top_k)

        correct = (y > 0) == (target > 0)
        num_correct += torch.sum(correct).item()
        num_total += len(target)

        zeros = torch.zeros(target.shape)
        ones = torch.ones(target.shape)

        filled = torch.where(x == 0, ones, zeros)
        filled_correct = filled * correct
        num_filled_total += torch.sum(filled)
        num_filled_correct += torch.sum(filled_correct)

        num_neg_ones_correct += torch.sum((target == -1) * filled)
        num_ones_correct += torch.sum((target == 1) * filled)

    print(f'accuracy: {num_correct / num_total:.4f}')
    print(f'accuracy on filled: {num_filled_correct / num_filled_total:.4f}')
    print(f'proportion -1: {prop_neg_one}')


if __name__ == '__main__':
    dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout=0.1)
    top_k = 100
    # complete_ballots = max_intersection_completion(dataset.x_list, top_k)
    # num_ballots = len(dataset)
    # num_val = num_ballots // 3
    # num_train = num_ballots - num_val
    # train_dataset, val_dataset = data.random_split(dataset, [num_train, num_val])
    # validate_baseline(val_dataset, top_k)
    # validate(single_mi_completion, val_dataset)
    candidates = [torch.tensor([1, 1, 1, 1, -1, -1, 1, -1]), torch.tensor([-1, -1, 1, 1, 1, -1, 1, 1]),
                  torch.tensor([-1, 1, 1, -1, -1, 1, -1, -1]), torch.tensor([1, 0, -1, 0, -1, -1, -1, 0])]
    ballot = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0])
    print(single_mi_completion(ballot, candidates, top_k=3))
