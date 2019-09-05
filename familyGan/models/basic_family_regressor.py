import torch
from sklearn.base import BaseEstimator
import numpy as np

import config


class BasicFamilyReg(BaseEstimator):
    def __init__(self, seed=42):
        self.seed = seed

    def add_random_gender(self, y_pred, coefs:float = None):
        coefs = np.random.choice([2, -2], size=len(y_pred)) if coefs is None else coefs
        y_pred = torch.tensor(y_pred)
        return y_pred.to('cpu').numpy() + coefs[:, np.newaxis, np.newaxis] * config.gender_direction[np.newaxis, :, :]
