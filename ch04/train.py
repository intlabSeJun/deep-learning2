# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common import config
# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
#===============================================
#config.GPU = True
#===============================================
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from skip_gram import SkipGram
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb


# 하이퍼파라미터 설정
window_size = 5 #맥략에 포함시킬 인접단어 범위 : 양옆5개?
hidden_size = 100 # hidden-layer 출력 벡터 수
batch_size = 100
max_epoch = 10

# 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
""" 거대한 말뭉치 - 10000종류의 단어와 단어수:929589. 
corpus : 929589개의 단어들을 id를 labeling 해줌 0~9999. 총 10000가지의 단어들
word_to_id : 10000가지의 단어에 id를 부어 0~9999
id_to_world : 10000개의 id에 단어를 부여 
"""
vocab_size = len(word_to_id) # 10000

contexts, target = create_contexts_target(corpus, window_size)
#print(contexts.shape, target.shape);exit(1)
"""
929589개의 단어가 window_size만큼 양옆으로 짤리고 나머지를 중심단어의 targets으로 보고, 
중심단어를 기준으로 양옆 5단어를 contexts로 본다. 
contexts.shape : (929579, 10) # 중심단어를 기준으로 양옆 10개의 단어들의 나열 되어 있음.
target.shape : (929579,)  # 중심단어가 나열되어 있음. 
"""
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# 모델 등 생성
model = CBOW(vocab_size, hidden_size, window_size, corpus)  # 인스턴스 생성, 모델생성. ( 인접단어로 중심단어 추론 )
# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 학습 시작
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 나중에 사용할 수 있도록 필요한 데이터 저장
word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
