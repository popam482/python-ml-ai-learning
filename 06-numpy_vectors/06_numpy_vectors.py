import numpy as np

#rank 1 array
a = np.random.randn(5)

print("\n", a)

print("\n", a.shape)

print("\n", a.T)

print("\n", np.dot(a,a.T))

a = np.random.randn(5, 1)
print("\n", a)

print("\n", a.T)

print(np.dot(a, a.T))