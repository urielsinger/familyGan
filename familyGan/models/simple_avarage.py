from familyGan.models.basic_family_regressor import BasicFamilyReg
import numpy as np
import config
import random


class SimpleAverageModel(BasicFamilyReg):
    def __init__(self, direction=config.age_kid_direction, coef=2, **kwargs):
        super().__init__(**kwargs)
        self.direction = direction
        self.coef = coef

    def fit(self, X_fathers, X_mothers, y_child):
        pass

    def predict(self, X_fathers, X_mothers):
        gender_direction = np.random.choice([2, -2], size=X_fathers.shape[0])
        if self.bgd_norm:
            latent_index = self.background_heuristic(X_fathers,X_mothers)
            y_pred_mean = np.mean([X_fathers, X_mothers], axis=0)
            y_pred_old = X_fathers * (~latent_index) + y_pred_mean * latent_index
        else:
            y_pred_old = np.mean([X_fathers, X_mothers], axis=0)
        y_pred_young = y_pred_old + self.coef * self.direction
        return self.add_random_gender(y_pred_young, coefs=gender_direction)
