import torch
from sklearn.base import BaseEstimator
import numpy as np

from familyGan import config


class BasicFamilyReg(BaseEstimator):
    def __init__(self, seed=42, **kwargs):
        self.seed = seed

    def add_random_gender(self, y_pred, coef:float = None):
        coefs = np.random.choice([2, -2], size=len(y_pred)) if coef is None else np.array([coef] * len(y_pred))
        y_pred = torch.tensor(y_pred)
        return y_pred.to('cpu').numpy() + coefs[:, np.newaxis, np.newaxis] * config.gender_direction[np.newaxis, :, :]
