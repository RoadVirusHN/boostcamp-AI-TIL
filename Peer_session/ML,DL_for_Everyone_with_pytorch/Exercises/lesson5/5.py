import torch

# 1. class 이용하여 모델 디자인
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]]) # Variable 함수는 deprecated 되었다.
# 대신 requires_grad를 True로 줌으로써 활용 가능

class Model(torch.nn.Module): # 이름은 아무거나 정할 수 있지만 여기서는 Model
    def __init__(self): # 기타 추가 element 생성 가능
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # 선형 모델 생성 y = x 그래프이므로 1, 1


    def forward(self, x): # x를 넣어준 뒤 예상한 y를 리턴 
        y_pred = self.linear(x)
        return y_pred

model = Model() # 모델 객체 생성

# 2. PyTorch API를 통하여 loss 함수와 optimizer 구현
criterion = torch.nn.MSELoss(size_average=False) # MSE Loss function 설정
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #  SGD 설정 lr = learning rate
# optimizer = torch.optim.Rprop(model.parameters(), lr=0.01) # loss값 6.5512e-12 결과값 4: 8.0000 ==> 최고의 성능! 
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01) # loss값 0.0003 결과값 4: 7.9787 
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01) # closure 함수를 요구함, 너무 깊게 들어가는 것 같으므로 생략
# optimizer = torch.optim.ASGD(model.parameters(), lr=0.01) # loss값 9.572e-05 결과값 4: 7.9887 
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.01) # loss값 0.2047 결과값 4: 7.4473 
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # loss값 0.0646 결과값 4: 7.691 
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01) # loss값 54.52 결과값 4: -0.4639

for epoch in range(500):
    y_pred = model(x_data)# x_data = metrics

    loss = criterion(y_pred, y_data) # 결과 값을 통해 loss 값 계산
    print(epoch, loss.item()) # 결과값이 0 dimension 이므로 data[0]를 할 수 없다는 결과가 나옴. loss.item()으로 확인가능

    optimizer.zero_grad() # 모든 gradient 값 0으로 초기화
    loss.backward() # back propagation
    optimizer.step() # back propagation 값으로 parameters 값을 업데이트해줌
    # 수많은 데이터에서는 비효율적인 방법이지만 지금 과정에서는 괜찮다.

# 3. 모델 테스트
hour_var =torch.Tensor([[4.0]]) # 학습을 tensor로 했으므로 input 또한 tensor로 줘야한다.
print("predict : ", 4, model.forward(hour_var).data[0][0])