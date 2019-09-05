from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL import Image


class Metric(ABC):
    @abstractmethod
    @classmethod
    def calculate_metric(cls, latent_true_mat: np.ndarray, latent_pred_mat: np.ndarray,
                         img_true_list: List[Image], img_pred_list: List[Image]) -> float:
        pass
