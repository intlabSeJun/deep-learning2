# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus): #10000, 100, 5, corpus
        V, H = vocab_size, hidden_size # 10000, 100

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f') # 10000, 100
        W_out = 0.01 * np.random.randn(V, H).astype('f') # 10000, 100

        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size): # 10
            layer = Embedding(W_in)  # Embedding 계층 사용, weight-sharing 통해서 10개 layer 생성.
            self.in_layers.append(layer) # 10개 layer 이어줌, 객체 리스트 [객체1, 객체2, ...]

        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치와 기울기를 배열에 모은다.
        layers = self.in_layers + [self.ns_loss] # 객체들의 묶음, 객체 리스트 [layer객체1, layer객체2, ..., Negativ객체] - 11개 객체

        self.params, self.grads = [], []
        for layer in layers: # 10개 layer, 1개 negative params,grads를 모음 [array(layer1_weight), array(layer2_weight2), ..., array(negativ_weight(w_out)]
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0

        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        """
        인접한 단어가 양옆 합해서 10개 이니까 layer을 10개 만들어 놓고 모두 weight-sharing 통해서 가중치를 정해 놓은 후에 
        임베딩 방식으로 random-batch를 한 contexts에서 인접단어들의 번호를 열단위로 받아와서 이를 W에 인덱싱하여 weight 값을 추출(batch,100)
        이렇게 추출한 인접한 단어들의 weight 값을 모두 더해줌.(왜 더해주지? - 아 모두 더해서 개수만큼 나누어 주었지!)
        """
        h *= 1 / len(self.in_layers) # 더한 개수만큼 나누어줌. (batch, weight_in_out) - (100,100)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
