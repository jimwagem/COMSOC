import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim_list, bottleneck_dim):
        super(Encoder, self).__init__()

        layers = []
        dims = [in_dim] + h_dim_list + [bottleneck_dim]
        # Construct list of linear layers + ReLU activations
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
        # Convert list to nn Sequential
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim_list, bottleneck_dim):
        super(Decoder, self).__init__()

        layers = []
        h_dim_list.reverse()
        dims = [bottleneck_dim] + h_dim_list + [in_dim]
        # Construct list of linear layers + ReLU activations
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
        # Convert list to nn Sequential
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, h_dim_list, bottleneck_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_dim, h_dim_list, bottleneck_dim)
        self.decoder = Decoder(in_dim, h_dim_list, bottleneck_dim)

    def forward(self, x):
        is_flat = (len(x.shape) == 1)
        if is_flat:
            # Add batch dimension
            x = x.unsqueeze(dim=0)
        z = self.encoder(x)
        x_rec = self.decoder(z)
        output = torch.tanh(x_rec)
        if is_flat:
            output = output.squeeze()
        return output

    # Use trained model to replace zeros in batch of ballots
    def complete_ballots(self, ballots):
        zeros = torch.zeros(ballots.shape)
        ones = torch.ones(ballots.shape)
        # Complete ballots
        completed = self(ballots)
        # transform floats to 1 / -1
        completed = torch.where(completed > 0, ones, -ones)
        # Mask with 1 where info is missing
        missing_mask = torch.where(ballots == 0, ones, zeros)
        # use only completed values that are actually missing
        fill_missing = missing_mask*completed
        # Add fill to ballots
        return ballots + fill_missing

if __name__ == '__main__':
    ae = AutoEncoder(10, [8], 6)
