import numpy as np
D, N = 8, 7
x = np.random.randn(1,D)
y = np.repeat(x,N,axis=0)
print(x)
print(y)

dy = np.random.randn(N,D)
dx = np.sum(dy, axis=-0, keepdims=True)
print(dx.shape)

dk = np.repeat(dx, N, axis=0)
print(dk.shape)