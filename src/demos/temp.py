import numpy as np

P = 20; N = 30
z = np.random.random(N*P*4)
print z

for ii in range(P):
    for jj in range(N):
        ind = ii*N*4 + jj*4 + 4
        z[ind-1] =  ii
print z
