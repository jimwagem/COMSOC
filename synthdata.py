import numpy as np
import torch
import torch.utils.data as data

class Project():
    def __init__(self, num_categories, difficulty=None, category_prior=None):
        # Difficulty is the probability that a voter is able to form an opinion on the project
        if difficulty is None:
            self.difficulty = torch.rand(1)
        else:
            self.difficulty = torch.tensor(difficulty)
        
        # For each category, decide where the project belongs
        # IDEA: difficulty per category
        if category_prior is None:
            category_prior = torch.tensor([0.5]*num_categories)
        self.categories = torch.bernoulli(category_prior)
    
    def understand(self):
        # IDEA: voter side difficulty
        return torch.bernoulli(self.difficulty) == 0

class Voter():
    """ Voter object. Has a preference ranging from 0 to 1 for each category."""

    # IDEA: include a voter ignorance parameter.
    def __init__(self, num_categories):
        # Preference[i] is the probability that the voter likes category i
        self.preferences = torch.rand(num_categories)
        self.num_categories = num_categories
    
    def approve(self, project):
        """A voter approves of a project if it agrees with the majority of categories"""
        proj_categories = project.categories
        agree_probs = proj_categories*self.preferences + (1-proj_categories)*(1-self.preferences)

        agreements = torch.bernoulli(agree_probs)
        return torch.mean(agreements) >= 0.5


class SynthDataLoader(data.Dataset):
    def __init__(self, num_categories, num_voters, num_projects, budget=2000000, total_cost=4500000):
        self.num_categories = num_categories
        self.num_voters = num_voters
        self.num_projects = num_projects
        self.n_projects = num_projects
        self.hold_election()

        # Decide project_costs.
        project_costs = np.random.uniform(0.01, 1, size=(num_projects))
        project_costs *= total_cost/np.sum(project_costs)
        self.project_costs = torch.Tensor(project_costs)
        self.budget = budget

    # Create x / target tensors from pabulib file
    def hold_election(self):
        self.voters = [Voter(self.num_categories) for _ in range(self.num_voters)]
        self.projects = [Project(self.num_categories, difficulty=0.25) for _ in range(self.num_projects)]
        zeros = torch.zeros(self.num_projects)
        ones = torch.ones(self.num_projects)
        self.x = []
        self.targets = []

        count=0
        for voter in self.voters:
            count += 1
            if count%100==0:
                print(f'created {count} voters')
            # Ballot is reported approval/disapproval
            # Expert ballot is where a voter understands all projects
            ballot = []
            expert_ballot = []
            for project in self.projects:
                if voter.approve(project):
                    opinion = 1
                else:
                    opinion = -1
                expert_ballot.append(opinion)
                if project.understand():
                    ballot.append(opinion)
                else:
                    # Voter does not understand project
                    ballot.append(0)
            ballot = torch.tensor(ballot)
            expert_ballot = torch.tensor(expert_ballot)

            self.targets.append(expert_ballot)
            self.x.append(ballot)

        self.x = torch.stack(self.x).float()
        self.targets = torch.stack(self.targets).float()

    def __getitem__(self, idx):
        return self.x[idx], self.targets[idx]

    def __len__(self):
        return self.x.shape[0]

if __name__ == '__main__':
    synthdata = SynthDataLoader(num_categories=3, num_voters=10, num_projects=5)
    train_loader = data.DataLoader(synthdata, batch_size=4)
    print(synthdata.project_costs)
    for x, target in train_loader:
        print(x)
        print(target)
        break