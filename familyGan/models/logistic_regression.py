from sklearn.linear_model import LogisticRegression

from familyGan.load_data import load_false_triplets
from familyGan.models.basic_family_regressor import BasicFamilyReg
import numpy as np

class LogisticRegressor(BasicFamilyReg):
    def fit(self, X_fathers, X_mothers, y_child):
        false_fathers, false_mothers, false_child = load_false_triplets(X_fathers, X_mothers, y_child, example_amount = len(X_fathers))
        X_true = np.concatenate([X_fathers, X_mothers, y_child], axis=-1)
        X_false = np.concatenate([false_fathers, false_mothers, false_child], axis=-1)

        X = np.concatenate([X_true, X_false], axis=0)
        y = np.concatenate([np.ones(len(X_fathers)), np.zeros(len(X_fathers))])

        self.clf = LogisticRegression().fit(X,y)
        self.coefs = self.clf._coef[len(X_fathers * 2)]

    def predict(self, X_fathers, X_mothers):
        pass
