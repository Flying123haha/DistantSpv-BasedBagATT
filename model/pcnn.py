import torch
import torch.nn as nn


class CNNwithPool(nn.Module):
    def __init__(self, cnn_layers, kernel_size):
        super(CNNwithPool, self).__init__()
        self.cnn = nn.Conv2d(1, cnn_layers, kernel_size)
        self.cnn.bias.data.copy_(nn.init.constant(self.cnn.bias.data, 0.))

    def forward(self, x, mask):

        cnn_out = self.cnn(x).squeeze(3)    # torch.Size([sequence num, 1, 82, 160]) -> torch.Size([sequence num, 230, 80])
        mask = mask[:, :, :cnn_out.size(2)].transpose(0,1)  # [65，3， 80]    65:sentence的数量，3: wf和两种pf， 80:句子长度
        pcnn_out, _ = torch.max(cnn_out.unsqueeze(1) + mask.unsqueeze(2), 3)    # torch.Size([65, 3, 230])

        return pcnn_out
