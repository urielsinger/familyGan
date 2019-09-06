from typing import List

import numpy as np
from PIL import Image

from metrics.base_metric import Metric


class RMSE(Metric):
    @classmethod
    def calculate_metric(cls, latent_true_mat: np.ndarray, latent_pred_mat: np.ndarray,
                         img_true_list: List[Image], img_pred_list: List[Image]) -> float:
        mse = ((latent_true_mat - latent_pred_mat) ** 2).mean(axis=0)
        return np.sqrt(mse)
