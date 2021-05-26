import numpy as np
import torch
import torch.utils.data as data

class Project():
    def __init__(self, num_categories, cost, category_prior=None):
        # Difficulty is the probability that a voter is able to form an opinion on the project
        
        self.cost=cost
        # For each category, decide where the project belongs
        # IDEA: difficulty per category
        if category_prior is None:
            category_prior = torch.tensor([0.5]*num_categories)
        self.categories = torch.bernoulli(category_prior)

class Voter():
    """ Voter object. Has a preference ranging from 0 to 1 for each category."""

    # IDEA: include a voter ignorance parameter.
    def __init__(self, num_categories):
        # Preference[i] is the probability that the voter likes category i
        self.preferences = torch.rand(num_categories)
        self.num_categories = num_categories
    
    def approve(self, project, num_samples=10, prior=0.5):
        """A voter approves of a project if it agrees with the majority of categories"""
        proj_categories = project.categories
        agree_probs = proj_categories*self.preferences + (1-proj_categories)*(1-self.preferences)

        # agreements = torch.bernoulli(agree_probs)
        agree_prob = torch.mean(agree_probs)
        agreements = torch.bernoulli(torch.stack(num_samples*[agree_prob]))
        return torch.mean(agreements) >= 0.5


class SynthDataLoader(data.Dataset):
    def __init__(self, num_categories, num_voters, num_projects, budget=2000000, total_cost=4500000, num_samples=11, prior=0.5, dropout = 0):
        self.num_categories = num_categories
        self.num_voters = num_voters
        self.num_projects = num_projects
        self.n_projects = num_projects

        # Decide project_costs.
        project_costs = np.random.uniform(0.01, 1, size=(num_projects))
        project_costs *= total_cost/np.sum(project_costs)
        self.project_costs = torch.Tensor(project_costs)
        self.budget = budget
        self.num_samples=num_samples
        self.prior=prior
        self.dropout = dropout
        self.hold_election()

    # Create x / target tensors from pabulib file
    def hold_election(self):
        self.voters = [Voter(self.num_categories) for _ in range(self.num_voters)]
        self.projects = [Project(self.num_categories, cost=self.project_costs[i]) for i in range(self.num_projects)]
        self.targets = []
        
        count = 0
        for voter in self.voters:
            count += 1
            if count%1000==0:
                print(f'Creating voter {count}')
            # Ballot is reported approval/disapproval
            # Expert ballot is where a voter understands all projects
            ballot = []
            expert_ballot = []
            for project in self.projects:
                if voter.approve(project, self.num_samples, self.prior):
                    opinion = 1.0
                else:
                    opinion = -1.0
                expert_ballot.append(opinion)
                ballot.append(opinion)
                
            ballot = torch.tensor(ballot)
            expert_ballot = torch.tensor(expert_ballot)

            self.targets.append(expert_ballot)

        self.targets = torch.stack(self.targets)
        self.create_x_from_dropout()
    
    def create_x_from_dropout(self):
        # Dropout some percent of x tensors
        # IDEA: Possibly have different dropout probabilities

        mask = torch.rand(self.targets.shape)
        zeros = torch.zeros(mask.shape)
        ones = torch.ones(mask.shape)
        mask = torch.where(mask < self.dropout, zeros, ones)
        self.x = self.targets * mask
    
    def __getitem__(self, idx):
        return self.x[idx], self.targets[idx]

    def __len__(self):
        return self.x.shape[0]

if __name__ == '__main__':
    synthdata = SynthDataLoader(num_categories=3, num_voters=10, num_projects=5)
    train_loader = data.DataLoader(synthdata, batch_size=4)
    for x, target in train_loader:
        print(x)
        print(target)
        break