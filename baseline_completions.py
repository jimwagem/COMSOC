import torch

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
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def get_nearest_neighbor(sorted_candidates, index):
    """finds a nearest neighbor ballot for a project corresponding to index"""
    for candidate in sorted_candidates:
        if candidate[0][index] != 0:
            return candidate
    raise MissingPreferenceError


def max_intersection_completion(incomplete_ballots):
    """completes a set of ballots using nearest neighbor with respect to the intersection score"""
    completed_ballots = []
    for ballot in incomplete_ballots:
        # possible nearest neighbors should exclude the ballot under consideration
        nn_candidates = [x for x in incomplete_ballots if not torch.equal(x, ballot)]
        # complete the ballot using the nearest neighbor's preference
        completed_ballot = single_mi_completion(ballot, nn_candidates)
        completed_ballots.append(completed_ballot)
    return completed_ballots


def single_mi_completion(ballot, nn_candidates):
    """completes a single ballot using a project-wise nearest neighbour approach based on the intersection score"""
    # identify the projects for which there is an incomplete preference
    missing_indices = torch.nonzero(torch.where(ballot == 0, 1, 0), as_tuple=True)[0].tolist()
    # compute max intersection scores and sort accordingly
    sorted_by_score = get_sorted_candidates(ballot, nn_candidates)
    completed_ballot = ballot.clone()
    for index in missing_indices:
        # for each missing project, find the nearest neighbor who has a completed preference on it
        nearest_neighbor = get_nearest_neighbor(sorted_by_score, index)
        completed_ballot[index] = nearest_neighbor[0][index]
    return completed_ballot


if __name__ == '__main__':
    dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout=0.25)
    dataset.load_file()
    complete_ballots = max_intersection_completion(dataset.x_list)
    print(complete_ballots)
