import torch
from scipy.linalg import orth
from torch.nn.modules.loss import _Loss
from torch.optim import Adam

import config
from familyGan.models.basic_family_regressor import BasicFamilyReg
from sklearn.linear_model import LinearRegression
import numpy as np
from torch import nn


class RegressorAndDirection(BasicFamilyReg):
    def __init__(self, epochs: int = 10, lr: float = 1e-4):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.lr = lr
        self.model = ChildNet().to(self.device)
        self.loss = ChildLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=lr)

    def fit(self, X_fathers, X_mothers, y_child):
        # X = torch.from_numpy(np.concatenate([X_fathers, X_mothers], axis=-1)).float().to(self.device)
        X_fathers = np2torch(X_fathers)
        X_mothers = np2torch(X_mothers)
        y = np2torch(y_child)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            output = self.model(X_fathers, X_mothers)
            loss = self.loss.forward(output, y)

            loss.backward()
            self.optimizer.step()

    def predict(self, X_fathers, X_mothers):
        X_fathers = np2torch(X_fathers)
        X_mothers = np2torch(X_mothers)
        with torch.no_grad():
            return self.model(X_fathers, X_mothers)


class ChildNet(nn.Module):
    def __init__(self, latent_size=18 * 512):
        super().__init__()
        self.father_attention = nn.Linear(latent_size, len(config.all_directions))
        self.mother_attention = nn.Linear(latent_size, len(config.all_directions))
        self.linear = nn.Linear(latent_size * 2, latent_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, *input):
        X_fathers, X_mothers = input
        all_directions = np2torch(config.all_directions.reshape(len(config.all_directions), -1))

        fathers_coefs = self.father_attention(X_fathers)
        mothers_coefs = self.mother_attention(X_mothers)

        moved_fathers = X_fathers + torch.mm(fathers_coefs, all_directions)
        moved_mothers = X_mothers + torch.mm(mothers_coefs, all_directions)

        output = self.linear(torch.cat([moved_fathers, moved_mothers], dim=1))
        return output


class ChildLoss(_Loss):

    def forward(self, input, target, hyper_plane=config.all_directions):
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
