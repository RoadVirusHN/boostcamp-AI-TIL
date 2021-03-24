import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.Tensor([[0.], [0.], [1.0], [1.0]])

import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        result = self.linear(x)
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var =torch.Tensor([[1.0]])
print("predict : ", 1, model.forward(hour_var).data[0][0])

hour_var =torch.Tensor([[7.0]])
print("predict : ", 7, model.forward(hour_var).data[0][0])