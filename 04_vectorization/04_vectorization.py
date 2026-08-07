import numpy as np
import time

a = np.array([1,2,3,4])
print(a)

a = np.random.rand(1_000_000)
b = np.random.rand(1_000_000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
print("Vectorized version: " + str((toc - tic)*1000) + " ms")

c = 0
tic = time.time()
for i in range(1_000_000):
    c+=a[i]*b[i]
toc = time.time()

print(c)
print("For loop: " + str((toc - tic)*1000) + " ms")