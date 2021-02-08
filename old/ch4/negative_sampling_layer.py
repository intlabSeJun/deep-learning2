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

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h # 벡터의 내적은 곱셈노드 + 덧셈노드 => 미분하면 하면 h, target_W 바뀜
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler: # 한 단어를 대상으로 확률분포를 만든다.
    """ 다중분류에서 이진분류로 만들기 위해서 sigmoid를 사용하는데 여기서 loss를 구할 때에 긍정적 예는 1에 가깝게 부정적 예는 0에 가깝게 학습 하기 위해서
    loss를 합산한다. 그러나 어휘수가 많아지면 부정적 예는 감당할 수 없기 때문에 '네거티브 셈플링'을 통해서 선별함.
    여기서는 sample_size만큼 부정적 단어를 선택하는 예제. 이를 통해 부정적 예 단어중 빈도수가 많은 것들을 확률적으로 높게 고를 수 잇게 하였고, 빈도수가
    적은 단어들이 무시되지 않도록 제곱근을 주어서 빈도수가 적은 단어도 선택 가능할 수 있게 함.
    어떤 문장이 있으면 해당 문장에서 단어들에게 ID를 부여할 수 있고 이는 중복 가능하다.
    ID 목록과, 제곱할 값, 부정적 예 샘플링 개수"""
    def __init__(self, corpus, power, sample_size): #단어 ID목록, 확률분포에 제곱할 값, '부정적 예 샘플링' 수행하는 횟수
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None # 확률 분포

        counts = collections.Counter()

        for word_id in corpus: # 어떤 단어가 몇번 나오는지 count
            counts[word_id] += 1


        vocab_size = len(counts) # 단어 종류 수

        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i] # word_p에 빈도수를 저장하고

        # 빈도수의 power만큼 제곱한 후에 확률값을 구해줌.
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target): # target 인수로 지정한 단어를 positive로 해석, 그 외의 단어 id를 샘플링
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32) # 샘플링 배열(결과)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]

                p[target_idx] = 0 # 정답데이터는 샘플링 안되게 함
                p /= p.sum() # 확률로 다시 만들어줌 (총합 1되게)

                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p) # 5종류중에 p에 따른 확률로 sample_size만큼 구함.
                # loss에 더할 네거티브 셈플링된 단어들의 집합. 각 행은 batch를 나타내고 해당 문장에서 네거티브 셈플링을 통해 단어를 선별함.
        else:
            # GPU(cupy）
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)] # 정답있는 레이어 포함해서 (샘플링 + 1)
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        # 매개변수, 기울기 한 곳에 넣음
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 긍정적인 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32) # 정답 레이블 = 1
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적인 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32) # negative 레이블 = 0
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i] # i번째 열벡터 가져오기
            score = self.embed_dot_layers[1 + i].forward(h, negative_target) # h는 모든 계층에 같은 값 줌
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore) # repeat node라서 다 더함

        return dh


#UnigramSampler 클래스 사용예제
corpus = np.array([0,1,2,3,4,1,2,3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus, power, sample_size)

target = np.array([1,3,0])
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)