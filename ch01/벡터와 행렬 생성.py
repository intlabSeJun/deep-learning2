import numpy as np

x = np.array([1,2,3])
class_x = x.__class__
print(f'x_class: {class_x}, x_shape:{x.shape}, x_dim:{x.ndim}\n{type(x)}')

W = np.array([[1,2,3],[4,5,6]])
print(f'W_shape:{W.shape}, W_ndim:{W.ndim}')