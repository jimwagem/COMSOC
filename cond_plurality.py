import torch
import numpy as np
import torch.utils.data as data
from dataloader import RealDataLoader
from synthdata import SynthDataLoader
from evaluation import evaluate_acc, evaluate_outcome

class Conditional_plurality():
    def __init__(self, train_dataset, n_projects, project_costs, use_costs=False):
        #Save vals
        self.n_projects = n_projects
        self.project_costs = project_costs
        self.use_costs = use_costs

        # Construct matrix
        # matrix is of size (n_projects, 2, n_projects)
        # matrix[x,b,y] is the sum of votes on project y, by people who voted b (0 for disapprove, 1 for approve) on project x
        self.matrix = torch.zeros(size=(self.n_projects, 2, self.n_projects))

        for i, (ballot, _) in enumerate(train_dataset):
            # if i%1000 == 0:
            #     print('did 1000')
            for x, vote in enumerate(ballot):
                if vote == 0:
                    continue
                elif vote == 1:
                    b = 1
                else:
                    b = 0
                self.matrix[x,b] += ballot

    def forward(self,ballot):
        missing_votes = (ballot == 0)
        plurality_ballot = torch.zeros(self.n_projects)
        
        # Weight by cost
        if self.use_costs:
            weights = self.project_costs
        else:
            weights = torch.ones(self.n_projects)
        
        for x, (vote, weight) in enumerate(zip(ballot, weights)):
            if vote == 0:
                    continue
            elif vote == 1:
                b = 1
            else:
                b = 0
            plurality_ballot += weight*self.matrix[x, b]
        
        completed_ballot = torch.zeros(self.n_projects)
        completed_ballot[~missing_votes] = ballot[~missing_votes]
        completed_ballot[missing_votes] = torch.where(plurality_ballot > 0, 1.0, -1.0)[missing_votes]
        return completed_ballot

    def __call__(self, ballot):
        return self.forward(ballot)
    
    def complete_ballots(self, ballots):
        completed_ballots = []
        for ballot in ballots:
            completed_ballots.append(self.forward(ballot))
        return torch.stack(completed_ballots)
        
def validate_evaluation(dataset, use_costs=False):
    """Validate using the model in evaluation.py"""
    n_projects = dataset.n_projects
    project_costs = dataset.project_costs
    num_ballots = len(dataset)
    num_val = num_ballots // 3
    num_train = num_ballots - num_val
    train_dataset, val_dataset = data.random_split(dataset, [num_train, num_val])
    model = Conditional_plurality(train_dataset, n_projects, project_costs, use_costs=use_costs)
    evaluate_acc(model, val_dataset=val_dataset)
    evaluate_outcome(model, dataset, val_dataset, is_function=True)
    
if __name__=="__main__":
    # dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout=0.25)
    dataset = SynthDataLoader(num_categories=5, num_voters=8000, num_projects=80, dropout=0.25, prior=0.6)
    validate_evaluation(dataset)
    
