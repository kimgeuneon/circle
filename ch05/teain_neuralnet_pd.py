# coding: utf-8
import sys
import numpy as np
from dataset.mnist import load_mnist
from three_layer_net import ImprovedFourLayerNet

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_loss_list = []
train_acc_list = []
test_acc_list = []

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 데이터 형태 맞추기 (TensorFlow 기반 네트워크에 맞춤)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# 데이터 확장 구성
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

# 선택한 네트워크 사용 - 예시로 ImprovedThreeLayerNet 사용
network = ImprovedFourLayerNet(input_size=784, hidden_size1=15, hidden_size2=15,hidden_size3=15, output_size=10)

# 학습 매개변수 설정
iters_num = 10000
batch_size = 100
learning_rate = 0.019


# 학습 시작
for i in range(iters_num):
    # 배치 데이터 생성
    batch = next(datagen.flow(x_train, t_train, batch_size=batch_size))
    x_batch, t_batch = batch
    
    # 오차 역전파로 기울기 계산
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]
    
    if i % 100 == 0:
        loss = network.loss(x_batch, t_batch)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Iter {i}, Loss: {loss:.4f} , Train_acc: {train_acc:.4f},Test_acc: {test_acc:.4f}")
