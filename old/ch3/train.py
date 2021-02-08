import sys
sys.path.append('..')
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from simple_skip_gram import SimpleSkipGram
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text) # corpus: 문장에서 단어의 label ('.'포함),

vocab_size = len(word_to_id) # 문장에서 단어 종류의 수, 7
contexts, target = create_contexts_target(corpus, window_size) # window_size만큼 단어들의 문맥, 단어 label

target = convert_one_hot(target, vocab_size) #(6,7) target을 vocab_size에 맞게 one_hot
contexts = convert_one_hot(contexts, vocab_size) # (6,2,7)
print(contexts);exit(1)

#model = SimpleSkipGram(vocab_size, hidden_size)
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.          fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 단어의 분산 표현 ( Win 저장 )
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
	print(word, word_vecs[word_id])