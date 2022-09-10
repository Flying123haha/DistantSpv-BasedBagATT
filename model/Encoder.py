import torch
import torch.nn as nn

class SAN(nn.Module):
    def __init__(self, word_dim, pos_dim, lam):
        super(SAN, self).__init__()
        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(2 * pos_dim + word_dim, 3 * word_dim)
        self.fc1_att = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2_att = nn.Linear(3 * word_dim, 3 * word_dim)
        self.lam = lam
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc1_att.weight)
        nn.init.xavier_uniform_(self.fc2_att.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc1_att.bias)
        nn.init.zeros_(self.fc2_att.bias)

    def forward(self, Xp, Xe):
        # embedding
        A = torch.sigmoid((self.fc1(Xe / self.lam)))
        X = A * Xe + (1 - A) * torch.tanh(self.fc2(Xp))
        # encoder
        A = self.fc2_att(torch.tanh(self.fc1_att(X)))
        P = torch.softmax(A, 1)
        X = torch.sum(P * X, 1)
        return X

