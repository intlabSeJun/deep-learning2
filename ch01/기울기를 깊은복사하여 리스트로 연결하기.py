import numpy as np

a = np.array([[1,2,3],[3,4,5]])
b = np.array([[5,2,1],[53,34,21]])
print(id(a), id(b))
grads = [np.zeros_like(a), np.zeros_like(b)]
grads[0][...] = a
grads[1][...] = b
print(id(grads[0]),id(grads))