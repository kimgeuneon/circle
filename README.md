## 인공지능 개론 Fashon mnist 과제
202004128 홍기현
202004186 김근언

Fashion_train_CNN -> 이를 통해 실행시킬 수 있습니다. 

### 변경사항

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Fashion MNIST 데이터셋 다운로드 및 로드
(x_train, t_train), (x_test, t_test) = fashion_mnist.load_data()

# 데이터의 shape을 (N, H, W)에서 (N, C, H, W)로 변환
x_train = np.expand_dims(x_train, axis=1)  # (60000, 28, 28) -> (60000, 1, 28, 28)
x_test = np.expand_dims(x_test, axis=1)    # (10000, 28, 28) -> (10000, 1, 28, 28)

텐서플로우의 keras 를 이용해 fashion_nmist 를 다운받았습니다. 
데이터의 shape이 원본에선 (N,H,W) 3차원이 었는데 C를 추가하여 4차원으로 입력하였습니다. 
Trainer 에서 evaluate_sample_num_per_epoch=500 으로 조정하였습니다. 

## 학습결과
logs 파일안의 traning_log.txt 파일에 로그가 있습니다.  
또한 

![스크린샷 2024-06-04 오전 1 04 28](https://github.com/kimgeuneon/circle-root/assets/127713112/8ae15381-97d1-4c88-abcd-e31e08e1d8d1)

그래프로 찍은 모습입니다! 
