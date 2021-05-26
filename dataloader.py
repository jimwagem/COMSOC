import numpy as np
import pandas as pd
import torch.utils.data as data
import torch


class Project():
    def __init__(self, line):
        words = line.split(';')
        self.id = int(words[0])
        self.cost = int(words[1])
        self.categories = words[2].split(',')
        self.votes = int(words[3])
        self.name = words[4]
        self.target = words[5]
        self.selected = bool(words[6])

    def __str__(self):
        return str(self.id) + ': ' + self.name


class RealDataLoader(data.Dataset):
    def __init__(self, filename, dropout=0):
        self.filename = filename
        self.dropout = dropout
        self.load_file()

    # Create x / target tensors from pabulib file
    def load_file(self):
        mode = 0  # META / PROJECTS / VOTES
        self.x = []
        self.targets = []
        self.projects = []
        self.project_ids = []
        self.budget = 0
        self.project_costs = []
        with open(self.filename, 'r', encoding='utf8') as f:
            counter = 0
            for line in f:
                line = line.strip('\n')
                words = line.split(';')
                # Switch modes
                if line == 'PROJECTS':
                    mode = 1
                    counter = 0
                elif line == 'VOTES':
                    mode = 2
                    counter = 0
                elif len(line) > 6 and line[:6] == "budget":
                    self.budget = int(line[7:])
                # Skip first line with labels
                elif counter == 1:
                    pass
                # Different modes
                elif mode == 0 and words[0] == 'budget':
                    self.budget = int(words[1])
                elif mode == 1:
                    p = Project(line)
                    self.projects.append(p)
                    self.project_ids.append(p.id)
                    self.project_costs.append(p.cost)
                elif mode == 2:
                    t = -torch.ones(len(self.projects))
                    for vote in line.split(';')[4].split(','):
                        t[self.project_ids.index(int(vote))] = 1
                    self.targets.append(t)

                counter += 1

        self.targets_list = self.targets
        self.targets = torch.stack(self.targets)
        self.create_x_from_dropout()

        self.n_projects = len(self.projects)
        self.project_costs = torch.Tensor(self.project_costs)

    def to_listed_version(self, tensor):
        tensor_list = []
        for i in range(tensor.size(0)):
            component_tensor = torch.clone(tensor[i])
            tensor_list.append(component_tensor)
        self.x_list = tensor_list

    def create_x_from_dropout(self):
        # Dropout some percent of x tensors
        # IDEA: Possibly have different dropout probabilities

        mask = torch.rand(self.targets.shape)
        zeros = torch.zeros(mask.shape)
        ones = torch.ones(mask.shape)
        mask = torch.where(mask < self.dropout, zeros, ones)
        self.x = self.targets * mask
        self.to_listed_version(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.targets[idx]

    def __len__(self):
        return self.x.shape[0]


if __name__ == '__main__':
    rdl = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout=0.25)
    train_loader = data.DataLoader(rdl, batch_size=4)
    for x, target in train_loader:
        print(x)
        print(target)
        break
