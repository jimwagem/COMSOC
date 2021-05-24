import torch
import itertools

def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    From itertools website"""
    s = list(iterable)
    return itertools.chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def greedy_approval(ballots, budget, project_costs):
    """ Ballot is (num voters, num projects).
    Returns the indices of approved projects."""
    num_voters, num_projects = ballots.shape
    assert len(project_costs) == num_projects
    project_votes = torch.sum(ballots, dim=0)
    sorted_votes, indices = torch.sort(project_votes, descending=True)

    spend_budget = 0
    accepted_projects = []
    for project, cost in zip(indices, project_costs[indices]):
        if spend_budget + cost <= budget:
            accepted_projects.append(project)
            spend_budget += cost
    return accepted_projects

def max_approval(ballots, budget, project_costs):
    """ Ballot is (num voters, num projects).
    Returns the indices of approved projects.
    Not very efficient"""
    num_voters, num_projects = ballots.shape
    assert len(project_costs) == num_projects
    project_votes = torch.sum(ballots, dim=0)

    best_approval = 0
    best_projects = None
    project_set = set(range(num_projects))
    for projects in powerset(project_set):
        cost = 0
        approval = 0
        for project in projects:
            cost += project_costs[project]
            approval += project_votes[project]
        if cost <= budget and approval >= best_approval:
            best_approval = approval
            best_projects = projects
    return best_projects


    return accepted_projects

def ballot_distance(ballot1, ballot2, L1=True):
    diff = ballot1 - ballot2
    if L1:
        return torch.mean(torch.abs(diff))
    else:
        return torch.mean(diff*diff)
