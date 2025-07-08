
import torch

import torch.nn.functional as F

from torch import nn

class SpectralMLP(nn.Module):
    def __init__(self, alone=False):
        super().__init__()

        modules = [
            nn.Linear(300, 300//2),
            nn.ReLU(),
            nn.Linear(300//2, 1),
            nn.Sigmoid()
        ]

        self.main = nn.Sequential(*modules)

    def forward(self, batch):
        spectrum, fluor_line, fluor_img = batch

        out = spectrum
        out = self.main(out)

        return out

class SpectralCNN(nn.Module):
    @classmethod
    def get_spectral_conv_block(cls):
        return (
            nn.Conv1d(1, 30, 30, 1),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(30, 10, 10, 1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )

    def __init__(self, alone=False):
        super().__init__()

        modules = [
            *self.get_spectral_conv_block(),
            nn.Flatten(),
        ]

        if alone:
            modules.append(nn.Linear(2620, 20))
            modules.append(nn.ReLU())
            modules.append(nn.Linear(20, 20))
            modules.append(nn.ReLU())
            modules.append(nn.Linear(20, 1))
            modules.append(nn.Sigmoid())

        self.main = nn.Sequential(*modules)

    def forward(self, batch):
        spectrum, fluor_line, fluor_img = batch

        out = spectrum.view(spectrum.shape[0], 1, -1)
        out = self.main(out)

        return out

class FluorMLP(nn.Module):
    def __init__(self, alone=False):
        super().__init__()

        modules = [
            nn.Linear(4, 8),
            nn.Tanh()
        ]

        if alone:
            modules.append(nn.Linear(8, 1))
            modules.append(nn.Sigmoid())

        self.main = nn.Sequential(*modules)

    def forward(self, batch):
        spectrum, fluor_line, fluor_img = batch

        out = fluor_line
        out = self.main(out)

        return out

class FluorCNN(nn.Module):
    @classmethod
    def get_conv_block(cls, in_size, out_size, kernel=3, stride=1):
        return (
            nn.Conv2d(in_size, out_size, kernel, stride),
            nn.Tanh(),
            nn.BatchNorm2d(out_size),
            nn.MaxPool2d(2),
        )

    def __init__(self, alone=False):
        super().__init__()

        modules = [
            *self.get_conv_block(4, 8),
            *self.get_conv_block(8, 16),
            *self.get_conv_block(16, 32),
            *self.get_conv_block(32, 64),
            nn.Flatten()
        ]

        if alone:
            linear_input = 64*2*2
            modules.append(nn.Linear(linear_input, 32))
            modules.append(nn.Tanh())
            modules.append(nn.Linear(32, 1))
            modules.append(nn.Sigmoid())

        self.main = nn.Sequential(*modules)

    def forward(self, batch):
        spectrum, fluor_line, fluor_img = batch

        out = fluor_img
        out = self.main(out)

        return out

class FusionNN_line(nn.Module):
    def __init__(self):
        super().__init__()

        self.spectral_cnn = SpectralCNN()
        self.fluor_mlp = FluorMLP()

        size = 2620+8

        self.linear1 = nn.Linear(size, size//2)
        self.linear2 = nn.Linear(size//2, 1)

        self.act_out = nn.Sigmoid()

    def forward(self, batch):
        spectral_out = self.spectral_cnn(batch)
        fluor_out = self.fluor_mlp(batch)

        out = torch.cat((spectral_out, fluor_out), axis=1)

        out = self.linear1(out)
        out = F.relu(out)

        out = self.linear2(out)
        out = self.act_out(out)

        return out

class FusionNN_img(nn.Module):
    def __init__(self):
        super().__init__()

        self.spectral_cnn = SpectralCNN()
        self.fluor_cnn = FluorCNN()

        size = 2620+256

        self.linear1 = nn.Linear(size, size//2)
        self.linear2 = nn.Linear(size//2, 1)

        self.act_out = nn.Sigmoid()

    def forward(self, batch):
        spectrum, fluor_line, fluor_img = batch

        spectral_out = self.spectral_cnn(batch)
        fluor_out = self.fluor_cnn(batch)

        out = torch.cat((spectral_out, fluor_out), axis=1)

        out = self.linear1(out)
        out = F.relu(out)
        
        out = self.linear2(out)
        out = self.act_out(out)

        return out
