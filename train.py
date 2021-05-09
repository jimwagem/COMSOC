import torch
import torch.nn as nn
import torch.utils.data as data

from autoencoder import AutoEncoder
from dataloader import RealDataLoader

def train():
    # Init dataset
    dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = 0.25)
    trainloader = data.DataLoader(dataset, batch_size=1, shuffle = True)

    # Init Autoencoder
    criterion = nn.MSELoss()
    model = AutoEncoder(dataset.n_projects, [], dataset.n_projects)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    losses = []
    for i in range(5):
        print('epoch', i)
        for j, (x, target) in enumerate(trainloader):
            if j % 2000 == 0 and j > 0:
                print('TRAIN LOSS:', torch.mean(torch.Tensor(losses)).item())
                losses = []
            optimizer.zero_grad()

            # Forward
            y = model.forward(x)
            loss = criterion(y, target)
            # print(torch.sum(torch.max(torch.zeros(y.shape), y)).item(), torch.sum(torch.max(torch.zeros(y.shape), target)).item())
            losses.append(loss.item())

            # backward
            loss.backward()
            optimizer.step()

    return model

if __name__ == '__main__':
    model = train()

    dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = 0.25)

    completed = model.complete_ballots(dataset.x)
    print(completed)
