import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
	def __init__(self, vocab_size, hidden_size):
		V, H = vocab_size, hidden_size

		# 가중치 초기화
		W_in = 0.01 * np.random.randn(V, H).astype('f')
		W_out = 0.01 * np.random.randn(H, V).astype('f')

		# 계층 생성
		# layer0, layer1은 weight-sharing
		self.in_layer0 = MatMul(W_in) ## 입력층은 윈도우 크기만큼 만들어야함, 인스턴스 생성.
		self.in_layer1 = MatMul(W_in)
		self.out_layer = MatMul(W_out)
		self.loss_layer = SoftmaxWithLoss()

		# 모든 가중치와 기울기를 리스트에 모음
		layers = [self.in_layer0, self.in_layer1, self.out_layer, self.loss_layer]
		self.params, self.grads = [], []
		for layer in layers:
			self.params += layer.params
			self.grads += layer.grads

		# 인스턴스 변수에 단어의 분산 표현 저장
		self.word_vecs = W_in
	
	def forward(self, contexts, target):
		# 양옆 단어에 대한 x*Win을 batch만큼 수행. -> 해당단어가 중심단어에 관해 어느정도의 의미가 있는지를 나타내(분산표현)
		# -> one_hot으로 표현되어 matmul이 수행되므로 weight에서 해당 행이 분산표현 벡터(값)이 됨.
		h0 = self.in_layer0.forward(contexts[:, 0]) # (batch, 7) * (vocab_size(7), hidden)
		h1 = self.in_layer1.forward(contexts[:, 1]) # (bathc, 7) * (vocab_size, hidden)
		h = (h0 + h1) * 0.5 # 양 옆의 분산표현의 합.
		score = self.out_layer.forward(h) # (batch,hidden) * ( hidden, vocab_size )
		# print(score)
		# print(target)
		loss = self.loss_layer.forward(score, target)
		return loss

	def backward(self, dout=1):
		ds = self.loss_layer.backward(dout)
		da = self.out_layer.backward(ds)
		da *= 0.5
		self.in_layer1.backward(da)
		self.in_layer0.backward(da)
		return None

		