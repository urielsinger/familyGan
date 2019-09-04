from sklearn.base import BaseEstimator


class BasicFamilyReg(BaseEstimator):
    def __init__(self, seed=42):
        self.seed = seed

