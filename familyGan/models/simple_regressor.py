from familyGan.models.basic_family_regressor import BasicFamilyReg
from sklearn.linear_model import LinearRegression
import numpy as np

class SimpleRegressor(BasicFamilyReg):
    def fit(self, X_fathers, X_mothers, y_child):
        self.model = LinearRegression()
        X = np.concatenate([X_fathers, X_mothers], axis=-1)
        self.model.fit(X, y_child)

    def predict(self, X_fathers, X_mothers):
        X = np.concatenate([X_fathers, X_mothers], axis=-1)
        return self.model.predict(X)