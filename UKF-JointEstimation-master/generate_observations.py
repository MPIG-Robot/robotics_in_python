import math
import numpy as np
from matplotlib import pyplot as pl

# generate observations segament
def obs_func(start_index, end_index, process_func, noise_func):
    size = end_index-start_index+1
    trues = [0]*size
    noisy = [0]*size
    for i in range(start_index, end_index+1):
        trues[i-start_index] = process_func(i)
        noisy[i-start_index] = trues[i-start_index] + noise_func(i)
    return trues, noisy

trues1, noisy1 = obs_func(0,500,lambda i: math.sin(i/20),lambda i: np.random.normal(0,1))
trues2, noisy2 = obs_func(501,1000,lambda i: math.sin(500/20)+(i-500)/100,lambda i: np.random.normal(0,1))
trues3, noisy3 = obs_func(1001,1500,lambda i: math.sin(500/20)+5+math.cos(i/20),lambda i: np.random.normal(0,1))

trues = trues1 + trues2 + trues3
noisy = noisy1 + noisy2 + noisy3

with open("observations-multi-model.txt",'w') as f:
    for noise in noisy:
        f.writelines(str(noise)+"\n")

pl.figure()
pl.plot(range(len(trues)), trues, 'r')
pl.scatter(range(len(noisy)), noisy, c='b')
pl.legend(['true','noise'])
pl.show()
