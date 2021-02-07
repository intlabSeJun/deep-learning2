import numpy as np
from common.util import *

def preprocess(text):
    text = text.lower() # 모두 소문자로
    text = text.replace('.',' .') #마침표를 고려함
    words = text.split(' ')#나눔

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus, word_to_id, id_to_word,sep='\n')

matrix = create_co_matrix(corpus, len(word_to_id), 1)
print(matrix)