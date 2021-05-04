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
        return self.id


class RealDataLoader(data.Dataset):
    def __init__(self, filename, dropout = 0):
        self.filename = filename
        self.mode = 0 # META / PROJECTS / VOTES
        self.dropout = dropout
        self.projects = []
        self.project_ids = []
        self.x = []
        self.targets = []
        with open(self.filename, 'r') as f:
            counter = 0
            for line in f:
                line = line.strip('\n')
                if line == 'PROJECTS':
                    self.mode = 1
                    counter = 0
                elif line == 'VOTES':
                    self.mode = 2
                    counter = 0
                elif counter == 1:
                    pass
                elif self.mode == 1:
                        p = Project(line)
                        self.projects.append(p)
                        self.project_ids.append(p.id)
                elif self.mode == 2:
                    t = -torch.ones(len(self.projects))
                    for vote in line.split(';')[4].split(','):
                        t[self.project_ids.index(int(vote))] = 1
                    self.targets.append(t)

                    # Dropout some percent
                    mask = torch.rand(len(self.projects))
                    zeros = torch.zeros(len(self.projects))
                    ones = torch.ones(len(self.projects))
                    mask = torch.where(mask < self.dropout, zeros, ones)

                    self.x.append(t * mask)


                counter += 1
        self.x = torch.stack(self.x)
        self.targets = torch.stack(self.targets)

    def __getitem__(self, idx):
        return self.x[idx], self.targets[idx]

    def __len__(self):
        return self.x.shape[0]

if __name__ == '__main__':
    rdl = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = 0.25)
    train_loader = data.DataLoader(rdl, batch_size=4)
    for x, target in train_loader:
        print(x)
        print(target)
        break
