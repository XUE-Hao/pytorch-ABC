import numpy as np
import torch
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        # 定义模型框架
        self.linear1 = nn.Linear(D_in, H, bias=False)
        self.linear2 = nn.Linear(H, D_out, bias=False)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred


model = TwoLayerNet(D_in, H, D_out)
loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4  # 1e-3 ~ 1e-4是Adam比较好的初始学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for it in range(5000):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(it, loss.item())

    loss.backward()

    optimizer.step()
