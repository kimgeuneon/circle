# coding: utf-8
import sys, os
print(sys.path)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from three_layer_net import ImprovedThreeLayerNetLeakyReLU
from three_layer_net import ImprovedThreeLayerNetReLu
from three_layer_net import ImprovedThreeLayerNetWithDropout
from three_layer_net import ImprovedThreeLayerNetGL
from three_layer_net import ImprovedFourLayerNet
from three_layer_net import FourLayerNet
from three_layer_net import ImprovedCNNThreeLayerNet,ImprovedThreeLayerNetLeakyReLU_Adam,ImprovedThreeLayerNetLeakyReLU_Adam_DropOUT



# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 데이터를 0과 1 사이로 스케일링
# x_train = x_train / 255.0
# x_test = x_test / 255.0


#network = FourLayerNet(input_size=784, hidden_size1=15, hidden_size2=15,hidden_size3=15,output_size=10)
#network = ImprovedFourLayerNet(input_size=784, hidden_size1=15, hidden_size2=15,hidden_size3=15,output_size=10)
#network = ImprovedThreeLayerNetGL(input_size=784, hidden_size1=15, hidden_size2=15,output_size=10)
#network = ImprovedThreeLayerNetReLu(input_size=784, hidden_size1=15, hidden_size2=15,output_size=10)
#network = ImprovedThreeLayerNetWithDropout(input_size=784, hidden_size1=15, hidden_size2=15,output_size=10)
#network = ImprovedThreeLayerNetLeakyReLU(input_size=784, hidden_size1=15, hidden_size2=15,output_size=10)
#network=ImprovedThreeLayerNetLeakyReLU_Adam(input_size=784, hidden_size1=15, hidden_size2=15,output_size=10) #이때 훈련 99 달성
network=ImprovedThreeLayerNetLeakyReLU_Adam_DropOUT(input_size=784, hidden_size1=15, hidden_size2=15,output_size=10)


iters_num = 10000 #총 반복 횟수
train_size = x_train.shape[0] #훈련 데이터의 총 샘플 수 60,0000
batch_size = 1200 #미니 배치의 크기
learning_rate = 0.019
#learning_rate_decay = 0.99  # 학습률 감소율 설정


train_loss_list = []
train_acc_list = []
test_acc_list = []


iter_per_epoch = max(train_size / batch_size, 1) #한 에폭당 필요한 반복 횟수
#iter_per_epoch = 100

log_file = open("training_log.txt", "w")

# 기존 print() 함수 저장
old_print = print

# 새로운 print 함수 정의
def new_print(*args, **kwargs):
    # 콘솔에 출력
    old_print(*args, **kwargs)
    # 파일에 기록
    old_print(*args, **kwargs, file=log_file)
    # 로그 파일 즉시 갱신
    log_file.flush()

# print 함수를 오버라이드
print = new_print


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식
    
    #확률적 경사 하강법 적용
    # for key in ('W1', 'b1', 'W2', 'b2','W3', 'b3'):
    #     network.params[key] -= learning_rate * grad[key]
    

    # Adam을 사용하여 매개변수 갱신
    network.optimizer.update(network.params, grad)
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    
    # if i % 100 == 0 or i == 9999 :
    #     loss = network.loss(x_batch, t_batch, train_flg=True)
    #     train_acc = network.accuracy(x_train, t_train, train_flg=True)
    #     test_acc = network.accuracy(x_test, t_test, train_flg=False)  # 테스트 시에는 드롭아웃 비활성화
    #     train_acc_list.append(train_acc)
    #     test_acc_list.append(test_acc)
    #     print(f"Iter {i}, Loss: {loss:.4f} , Train_acc: {train_acc:.4f}, Test_acc: {test_acc:.4f}")
    
    #에폭당 정확도와 로스값출력
    if i % iter_per_epoch == 0:
        loss = network.loss(x_batch, t_batch, train_flg=True)
        train_acc = network.accuracy(x_train, t_train,train_flg=True)
        test_acc = network.accuracy(x_test, t_test,train_flg=False)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'Epoch {i // iter_per_epoch}: Loss: {loss:.4f} Train Accuracy: {train_acc*100:.4f}%, Test Accuracy: {test_acc*100:.4f}%')
        if train_acc*100>99 and test_acc*100>95:break

# 로그 파일 닫기
log_file.close()

# print 함수를 원래대로 복구
print = old_print

from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    """Axis value to percentage formatter."""
    return f"{100 * y:.2f}%"  # 백분율 표시를 위한 형식 지정

formatter = FuncFormatter(to_percent)
epochs = range(len(train_acc_list))
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_list[:len(epochs)], label='Train Loss')  # train_loss_list 길이 조정
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_list, label='Train Accuracy')
plt.plot(epochs, test_acc_list, label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.gca().yaxis.set_major_formatter(formatter)  # y축을 백분율로 포맷팅
plt.legend()

plt.tight_layout()
plt.show()
