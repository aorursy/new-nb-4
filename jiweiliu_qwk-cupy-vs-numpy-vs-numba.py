import os

GPU_id = 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
import warnings

warnings.filterwarnings("ignore")



from numba import jit 

import numpy as np

from sklearn.metrics import cohen_kappa_score, confusion_matrix

import cupy as cp

import time

import matplotlib.pyplot as plt
def cupy_hist_int(x,n):

    bins = cp.arange(n+1)-0.5

    hist,_ = cp.histogram(x,bins=bins)

    return hist



def cupy_confusion_matrix(true,pred,n):

    cf = true*n+pred

    cf = cupy_hist_int(cf,n*n)

    return cf.reshape([n,n])



def cupy_quadKappa(act,pred,n=4,hist_range=(0,3)):

    act = cp.asarray(act,dtype=cp.int32)

    pred = cp.asarray(pred,dtype=cp.int32)

    O = cupy_confusion_matrix(act,pred,n)

    O = cp.divide(O,cp.sum(O))

    

    W = cp.zeros((n,n))

    for i in range(n):

        for j in range(n):

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    act_hist = cupy_hist_int(act,n)

    prd_hist = cupy_hist_int(pred,n)

    

    E = cp.outer(act_hist,prd_hist)

    E = cp.divide(E,cp.sum(E))

    

    num = cp.sum(cp.multiply(W,O))

    den = cp.sum(cp.multiply(W,E))

        

    return 1-np.divide(num,den)
def quadKappa(act,pred,n=4,hist_range=(0,3)):

    

    O = confusion_matrix(act,pred)

    O = np.divide(O,np.sum(O))

    

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E))

    

    num = np.sum(np.multiply(W,O))

    den = np.sum(np.multiply(W,E))

        

    return 1-np.divide(num,den)
@jit

def qwk3(a1, a2, max_rat=3):

    assert(len(a1) == len(a2))

    a1 = np.asarray(a1, dtype=np.int32)

    a2 = np.asarray(a2, dtype=np.int32)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e
size = 1000000

a = np.random.randint(0, 4, size)

p = np.random.randint(0, 4, size)

a.size, p.size

quadKappa(a,p)

qwk3(a,p)

qwk3(a,p)

cupy_quadKappa(a,p)

cupy_quadKappa(a,p)
cupy_time = []

numpy_time = []

numba_time = []

for i in range(5,9):

    size = 10**i

    a = np.random.randint(0, 4, size)

    p = np.random.randint(0, 4, size)

    

    start = time.time()

    quadKappa(a,p)

    numpy_time.append(time.time()-start)

    

    start = time.time()

    cupy_quadKappa(a,p)

    cupy_time.append(time.time()-start)

    

    start = time.time()

    qwk3(a,p)

    numba_time.append(time.time()-start)
plt.figure(figsize=(15,5))

colors = ['b','g','r']

xs = [10**i for i in range(5,9)]

plt.yscale('log')

plt.xlim(5*10**4,5*10**8)

plt.ylim(10**(-5),10**3)

plt.xscale('log')

plt.xlabel('number of sample')

plt.ylabel('run time: seconds')

plt.grid()



plt.scatter(xs,numpy_time,c='b',label='numpy')  

plt.scatter(xs,cupy_time,c='g',label='cupy') 

plt.scatter(xs,numba_time,c='r',label='numba') 



plt.plot(xs,numpy_time,c='b')  

plt.plot(xs,cupy_time,c='g') 

plt.plot(xs,numba_time,c='r') 

plt.legend(loc='upper left')

plt.title('qwk: cupy vs numpy vs numba')