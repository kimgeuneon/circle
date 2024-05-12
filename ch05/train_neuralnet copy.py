# coding: utf-8
import sys, os
print(sys.path)

import numpy as np
from dataset.mnist import load_mnist
from three_layer_net import ImprovedThreeLayerNet
from three_layer_net import ImprovedThreeLayerNetWithDropout
from three_layer_net import ImprovedThreeLayerNetGL
from three_layer_net import ImprovedFourLayerNet
from three_layer_net import FourLayerNet
from three_layer_net import ImprovedCNNThreeLayerNet


# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 데이터를 (배치 크기, 1, 28, 28) 형태로 변환
x_train = x_train.reshape(-1, 1, 28, 28).astype('float32') / 255
x_test = x_test.reshape(-1, 1, 28, 28).astype('float32') / 255

# 라벨 데이터는 원-핫 인코딩이 필요하다면 변환
# from keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)


#network = FourLayerNet(input_size=784, hidden_size1=15, hidden_size2=15,hidden_size3=15,output_size=10)
#network = ImprovedFourLayerNet(input_size=784, hidden_size1=15, hidden_size2=15,hidden_size3=15,output_size=10)
#network = ImprovedThreeLayerNetGL(input_size=784, hidden_size1=15, hidden_size2=15,output_size=10)
#network = ImprovedThreeLayerNet(input_size=784, hidden_size1=15, hidden_size2=15,output_size=10)
#network = ImprovedThreeLayerNetWithDropout(input_size=784, hidden_size1=15, hidden_size2=15,output_size=10)
# conv_params 설정: 필터 수, 필터 크기, 패딩, 스트라이드
conv_params = {
    'filter_num': 30,        # 필터의 수
    'filter_size': 10,        # 필터의 크기 (5x5)
    'pad': 0,                # 패딩
    'stride': 2              # 스트라이드
}

# 네트워크 초기화
network = ImprovedCNNThreeLayerNet(
    input_dim=(1, 28, 28),     # 입력 이미지 차원 (채널, 높이, 너비)
    conv_params=conv_params,   # 컨볼루션 계층 파라미터
    hidden_size1=15,           # 첫 번째 히든 계층 크기
    hidden_size2=15,           # 두 번째 히든 계층 크기
    output_size=10             # 출력 계층 크기 (클래스 수)
)


iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.019
#learning_rate_decay = 0.99  # 학습률 감소율 설정


train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
#iter_per_epoch = 100

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2','W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(i,train_acc, test_acc)
        #학습률 감소
        #learning_rate *= learning_rate_decay
