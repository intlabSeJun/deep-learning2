# coding: utf-8
from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)  # x에 대한 역전파(기울기)
        dW = np.dot(self.x.T, dout) # w에 대한 역전파(기울기)
        self.grads[0][...] = dW # w에 대한 기울기를 모아둠 ( 깊은 복사 )
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T) #(30,10)
        dW = np.dot(self.x.T, dout) #(10,3)
        db = np.sum(dout, axis=0) # 열마다 더함. @@더해줄때 앞의(sooftmax.backward) batch_size로 나누어줌으로써 1이 넘지 않게됨
       # print(f'w.shape:{W.shape}, x.shape:{self.x.shape},dx.shape:{dx.shape}, dW.shape:{dW.shape}');exit(1)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        #print(f'grads:{self.grads[0].shape},{self.grads[1].shape}');exit(1)
        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        #print(f'[forward]\nt.shape:{self.t.shape}')
        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1) # 예측한 인덱스만 저장됨.
            #print(self.t)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        #print(f'[softmax_backward]\nt.shape:{self.t.shape[0]}, dx.shape:{self.y.copy().shape}');exit(1)
        dx = self.y.copy() # softmax 통과한 값들 copy
        dx[np.arange(batch_size), self.t] -= 1 # softmax 계층의 역전파 수행.(y1-t1, y2-t2, y3-t3)
        dx *= dout # 합성함수의 연쇄법칙 수행.
        dx = dx / batch_size # 평균해준다 -> @@발산 막는 용도? / softmaxWithLoss이기 떄문에 이미 forward에서 cross_entropy로 loss를 더해주었기 때문에 bathc만큰 나누어 줌으로써 평균을 가지도록 하는듯
       # print(f'dx.shape:{dx.shape}');exit(1)

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout): # dout.shape : (30,3)
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class Dropout:
    '''
    http://arxiv.org/abs/1207.0580
    '''
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Embedding:
    def __init__(self, W): # 10000,100 random 가중치
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx] # shape:(100,100)

        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None
