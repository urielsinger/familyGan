import torch
from torch.nn.modules.loss import _Loss

import config
from familyGan.models.basic_family_regressor import BasicFamilyReg
from sklearn.linear_model import LinearRegression
import numpy as np
from torch import nn


class RegressorAndDirection(BasicFamilyReg):
    def fit(self, X_fathers, X_mothers, y_child):
        X = np.concatenate([X_fathers, X_mothers], axis=-1)

        self.model.fit(X, y_child)

    def predict(self, X_fathers, X_mothers):
        X = np.concatenate([X_fathers, X_mothers], axis=-1)
        return self.model.predict(X)


class ChildNet(nn.Module):
    def __init__(self, latent_size=18 * 512):
        self.linear = nn.Linear(latent_size * 2, latent_size)
        self.attention = nn.Linear(latent_size, len(config.all_directions))

    def forward(self, input):
        coefs = self.attention(input)
        output = input + torch.mm(coefs, config.all_directions)
        output = self.linear(output)
        return output


class ChildLoss(_Loss):
    def forward(self, input, target):
        hyper_plane = config.all_directions[np.newaxis, :, :].reshape(1, len(config.all_directions),
                                                                      -1) + target.reshape(target.shape[0], 1,
                                                                                          target.shape[1])
        hyper_plane = hyper_plane  # TODO transform to orthogonal
        pv = np.sum(hyper_plane * input[:, np.newaxis, :], axis=-1, keepdims=True)
        vv = np.sum(hyper_plane**2, axis=-1, keepdims=True)
        proj = input[:,np.newaxis,:]*(pv/vv)
        return np.linalg.norm(input - np.sum(proj, axis=1), axis=-1)
