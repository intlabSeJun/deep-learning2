import numpy as np

a = np.random.randn(1,5)
print(a, a.shape)

b = np.repeat(a,5,axis=1)
print(b, b.shape)