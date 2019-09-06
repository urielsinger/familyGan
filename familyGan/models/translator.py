import torch
from torch.nn.modules.loss import MSELoss
from torch.optim import Adam

from models.basic_family_regressor import BasicFamilyReg
from torch import nn


class Translator(BasicFamilyReg):
    def __init__(self, epochs: int = 10, lr: float = 1, coef: float = -2):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.lr = lr
        self.coef = coef
        self.model = ChildNet().to(self.device)
        self.loss = MSELoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=lr)

    def fit(self, X_fathers, X_mothers, y_child):
        # X = torch.from_numpy(np.concatenate([X_fathers, X_mothers], axis=-1)).float().to(self.device)
        X_fathers = np2torch(X_fathers)
        X_mothers = np2torch(X_mothers)
        y = np2torch(y_child)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            output = self.model(X_fathers, X_mothers)
            loss = self.loss.forward(output, y)

            loss.backward()
            self.optimizer.step()
            print("loss", loss)

    def predict(self, X_fathers, X_mothers):
        X_fathers = np2torch(X_fathers)
        X_mothers = np2torch(X_mothers)
        with torch.no_grad():
            y_pred = self.model(X_fathers, X_mothers)
        # y_pred = y_pred + self.coef * np2torch(config.age_kid_direction)
        return self.add_random_gender(y_pred)


class ChildNet(nn.Module):
    def __init__(self, latent_size=18 * 512):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.translation = torch.nn.Parameter(torch.randn(1, latent_size))
        self.register_parameter('translation', self.translation)

    def forward(self, *input):
        X_fathers, X_mothers = input
        X_parents = torch.mean(torch.stack([X_fathers, X_mothers]), dim=0)
        X_fathers_moved = X_parents.flatten(1, 2) + self.translation
        y_pred = X_fathers_moved.reshape(X_fathers.shape)
        return y_pred


def np2torch(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.from_numpy(a).float().to(device)
