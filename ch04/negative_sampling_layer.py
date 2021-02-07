# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx): # h, target
        # h.shape : (100,100), target.shape : (100,1)
        target_W = self.embed.forward(idx) # (100,100)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()

        for word_id in corpus: # corpus 단어들의 종류를 count하여 딕셔너리형태로 저장하는 식.
            counts[word_id] += 1 # courpus는 모든 단어들을 순서대로 labeling(숫자들로) 되어 있고, 이를 labeling-key, 개수-value로 하는 딕셔너리를 생성


        vocab_size = len(counts) # 10000개

        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size): # 단어의 종류마다 몇개가 있는지 word_p에 저장함.
            self.word_p[i] = counts[i]
        #print(len(self.word_p)) #10000

        # 원래 확률이 낮은 단어의 확률을 살짝 높이는 방법.
        self.word_p = np.power(self.word_p, power) # power 계수만큼 제곱.
        self.word_p /= np.sum(self.word_p) # 총합 1인 확률분포를 만듬

    def get_negative_sample(self, target):

        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32) #100,5

            for i in range(batch_size):
                p = self.word_p.copy()
                #print(len(p), p, sep='\n');exit(1)
                target_idx = target[i]
                p[target_idx] = 0 # target을 제외하고 각 단어의 종류마다 확률로 나타낸 p를 모든 확률의 합으로나누어줌
                p /= p.sum() # 해당 target 제외하고 나누어 주었기 때문에 target을 제외하고 나머지 확률이 나옴, target=0
                # 부정적인 예를 sample_size만큼 추출하는 것이구나!
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
                #print(negative_sample[i, :]);exit(1)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss: #w_out
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size # 상위 5개만
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        #print(len(self.embed_dot_layers));exit(1)
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params # array형식으로 params를 묶어서 나열. [np.array(w), np.array(w), ...]
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target) # target을 제외하고 나머지에서 뽑은 negative-sample
        #print(negative_sample.shape);exit(1) #(batch_size, sample_size)
        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh
