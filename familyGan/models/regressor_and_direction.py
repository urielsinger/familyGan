import random

import torch
from scipy.linalg import orth
from torch.nn.modules.loss import _Loss, MSELoss
from torch.optim import Adam

import config
from familyGan.models.basic_family_regressor import BasicFamilyReg
import numpy as np
from torch import nn


class RegressorAndDirection(BasicFamilyReg):
    def __init__(self, epochs: int = 10, lr: float = 1, coef: float = -1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.lr = lr
        self.coef = coef
        self.model = ChildNet().to(self.device)
        self.loss = MSELoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=lr)
        self.gamma = 10

    def fit(self, X_fathers, X_mothers, y_child):
        # X = torch.from_numpy(np.concatenate([X_fathers, X_mothers], axis=-1)).float().to(self.device)
        X_fathers = np2torch(X_fathers)
        X_mothers = np2torch(X_mothers)
        y = np2torch(y_child)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            output = self.model(X_fathers, X_mothers)
            loss = self.loss.forward(output, y) #+ self.gamma * torch.norm(self.model.bias)

            loss.backward()
            self.optimizer.step()
            print(loss)

    def predict(self, X_fathers, X_mothers):
        X_fathers = np2torch(X_fathers)
        X_mothers = np2torch(X_mothers)
        with torch.no_grad():
            y_pred = self.model(X_fathers, X_mothers)
        y_pred = y_pred + self.coef * np2torch(config.age_kid_direction)
        return self.add_random_gender(y_pred)


class ChildNet(nn.Module):
    def __init__(self, latent_size=18 * 512):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.father_weights = torch.nn.Parameter(torch.ones(1, latent_size) / 2)
        self.mother_weights = torch.nn.Parameter(torch.ones(1, latent_size) / 2)
        self.bias = torch.nn.Parameter(torch.randn(1, latent_size))

        self.register_parameter('father_params', self.father_weights)
        self.register_parameter('mother_params', self.mother_weights)
        self.register_parameter('bias', self.bias)

    def forward(self, *input):
        X_fathers, X_mothers = input
        X_fathers_moved = X_fathers.flatten(1, 2) * self.father_weights
        X_mothers_moved = X_mothers.flatten(1, 2) * self.mother_weights
        X_mean = torch.mean(torch.stack([X_fathers_moved, X_mothers_moved]), axis=0)
        # output = X_mean + self.bias

        y_pred = X_mean.reshape(X_fathers.shape)
        return y_pred


class ChildLoss(_Loss):

    def forward(self, input, target, hyper_plane=config.all_directions):
        input = input.flatten(1, 2)
        target = target.flatten(1, 2)
        hyper_plane = hyper_plane.reshape(hyper_plane.shape[0], -1)  # flatten
        hyper_plane = orth(hyper_plane.T).T  # orthogonize
        hyper_plane = np2torch(hyper_plane)

        # projection of the input on the hyper planes of y_true
        new_p = input - target
        pv = torch.sum(hyper_plane * new_p[:, np.newaxis, :], dim=-1, keepdim=True)
        vv = torch.sum(hyper_plane ** 2, dim=-1, keepdim=True)
        proj = hyper_plane * (pv / vv)
        losses = torch.norm(new_p - torch.sum(proj, dim=1), dim=-1)
        return losses.sum()

        # hyper_plane = hyper_plane[np.newaxis, :, :].reshape(1, len(hyper_plane), -1) + target.reshape(target.shape[0],
        #                                                                                               1,
        #                                                                                               target.shape[1])


def np2torch(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.from_numpy(a).float().to(device)
