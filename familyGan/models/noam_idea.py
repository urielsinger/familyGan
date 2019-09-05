from familyGan.models.basic_family_regressor import BasicFamilyReg
import torch

class NoamModel(BasicFamilyReg):
    def fit(self, X_fathers, X_mothers, y_child):
        father_weigh


def np2torch(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.from_numpy(a).float().to(device)
