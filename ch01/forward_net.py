# coding: utf-8
import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화
        W1 = np.random.randn(I, H) # 2,4
        b1 = np.random.randn(H) # 4
        W2 = np.random.randn(H, O) # 4,3
        b2 = np.random.randn(O) # 3

        # 계층 생성  ->  인스턴스를 생성하여 메모리할당
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 모든 가중치를 리스트에 모은다.
        self.params = []
        for i,layer in enumerate(self.layers):
            print(f'address', id(layer))
            print(f'count :{i}')
            self.params += layer.params
        print(self.params, len(self.params), (np.array(self.params).shape))
        for i in range(4):
            print(self.params[i])
            print(1)


    def predict(self, x):
        for layer in self.layers: #인스턴스를 불러옴.
            print('layer',(layer))
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x) # (10,3)


