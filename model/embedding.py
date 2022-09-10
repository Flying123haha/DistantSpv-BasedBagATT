import torch
import torch.nn as nn


class getEmbeddings(nn.Module):
    def __init__(self, word_size, word_length, feature_size, feature_length, Wv, pf1, pf2):
        super(getEmbeddings, self).__init__()
        self.x_embedding = nn.Embedding(word_length, word_size, padding_idx=0)
        self.ldist_embedding = nn.Embedding(feature_length, feature_size, padding_idx=0)
        self.rdist_embedding = nn.Embedding(feature_length, feature_size, padding_idx=0)
        self.x_embedding.weight.data.copy_(torch.from_numpy(Wv))
        self.ldist_embedding.weight.data.copy_(torch.from_numpy(pf1))
        self.rdist_embedding.weight.data.copy_(torch.from_numpy(pf2))

    def forward(self, x, ldist, rdist, leftEnt, rightEnt):
        x_embed = self.x_embedding(x)   ## [118,82] -> [118, 82, 50]
        ldist_embed = self.ldist_embedding(ldist) ## ## [118,82] -> [118, 82, 5]
        rdist_embed = self.rdist_embedding(rdist)
        xEnt_embed = self.word_ent_embedding(x_embed, leftEnt, rightEnt)
        Xp = torch.cat([x_embed, ldist_embed, rdist_embed], x_embed.dim() - 1)    ## [118,82, 160(50*3 +10)]
        Xe = torch.cat([x_embed, xEnt_embed], x_embed.dim() - 1)
        return Xp.unsqueeze(1), Xe

    def word_ent_embedding(self, X, X_Ent1, X_Ent2):

        X_Ent1 = self.x_embedding(X_Ent1).unsqueeze(1).expand(X.shape)  # [54] ->  [54, 82, 50]
        X_Ent2 = self.x_embedding(X_Ent2).unsqueeze(1).expand(X.shape)
        return torch.cat([X_Ent1, X_Ent2], -1)