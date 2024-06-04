# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.optimizer import *

import numpy as np

class Trainer:
    """Neural Network Trainer"""

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # Optimizer 설정
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        # 로그 파일 경로 설정
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file_path = os.path.join(log_dir, "training_log.txt")
        
        # 로그 파일 열기
        self.log_file = open(self.log_file_path, "w")

    def __del__(self):
        # 로그 파일 닫기
        self.log_file.close()

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        if self.verbose:
            print(f"Train loss: {loss}")

        # 로그 파일에 기록
        self.log_file.write(f"Train loss: {loss}\n")
        self.log_file.flush()  # 파일에 즉시 기록

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test

            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(f"=== epoch: {self.current_epoch}, train acc: {train_acc}, test acc: {test_acc} ===")
            
            # 로그 파일에 기록
            self.log_file.write(f"=== epoch: {self.current_epoch}, train acc: {train_acc}, test acc: {test_acc} ===\n")
            self.log_file.flush()  # 파일에 즉시 기록

        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()