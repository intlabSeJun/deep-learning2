import numpy as np

# 밑딥3 p172
words = ['you', 'say', 'goodbye', 'i', 'hello', '.']
p = [0.5,0.1, 0.05, 0.2, 0.05, 0.1]
a = np.random.choice(words, p=p, size=3)
print(a)

#0.75 제곱하여 확률이 낮은 단어의 확률 조금 높임
p = [0.7, 0.29, 0.01]
new_p = np.power(p,0.75)
new_p /= np.sum(new_p)
print(new_p)