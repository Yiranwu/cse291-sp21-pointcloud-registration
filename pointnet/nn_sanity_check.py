import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(data_dim, 1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,label_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_epochs=100
num_instances=64
data_dim=1024
label_dim=10
data = torch.rand([num_instances,data_dim])
label = torch.rand([num_instances,label_dim])
print('data=', data)
print('label=', label)
dataset = TensorDataset(data, label)
loader  = DataLoader(dataset, batch_size=2)
net = LinearNet()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()
for epoch in range(num_epochs):
    losses = []
    for data, label in loader:
        optimizer.zero_grad()
        pred = net(data)
        loss = loss_func(pred, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print('epoch %d, loss %f'%(epoch, sum(losses)/len(losses)))
