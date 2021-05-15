import numpy as np
import torch

from dataloader import RealDataLoader


def intersection_score(ballot, candidate):
    """determines the number of projects that two ballots both approve of"""
    return torch.sum(torch.where((ballot == 1) & (candidate == 1), 1, 0)).item()


def get_nearest_neighbor(ballot, candidates, missing_indices):
    """returns the ballot that has the largest intersection score with respect to a given ballot"""
    scores = []
    for candidate in candidates:
        # exclude candidates with missing preferences on the relevant entries
        if 0 in candidate:
            scores.append(0)
        else:
            scores.append(intersection_score(ballot, candidate))
    return candidates[np.argmax(np.array(scores))]


def max_intersection_completion(incomplete_ballots):
    """completes a set of ballots using nearest neighbor with respect to the intersection score"""
    completed_ballots = []
    for ballot in incomplete_ballots:
        # possible nearest neighbor should exclude the ballot under consideration
        nn_candidates = [x for x in incomplete_ballots if not torch.equal(x, ballot)]
        missing_indices = torch.nonzero(torch.where(ballot == 0, 1, 0), as_tuple=True)[0].tolist()
        nearest_neighbor = get_nearest_neighbor(ballot, nn_candidates, missing_indices)

        # complete the ballot using the nearest neighbor's preference
        completed_ballot = ballot.numpy()
        completed_ballot[missing_indices] = nearest_neighbor[missing_indices]
        completed_ballots.append(completed_ballot)
    return completed_ballots


if __name__ == '__main__':
    dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout=0.25)
    dataset.load_file()
    complete_ballots = max_intersection_completion(dataset.x_list)
    print(complete_ballots)
