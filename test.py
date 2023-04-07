import numpy as np
import random
np.random.seed(10)
random.seed(10)
import keras

def non_zero_replacer(num):
  if (num == 1.0):
    return float(np.random.uniform(low=0.0, high=0.5, size=1)[0])
  else:
    return 0.0
def block_noise_adder(min_block_size, max_block_size,mat):
  size = np.shape(mat)[0]
  n = np.shape(mat)[-1]
  for i in range(0,n):
    block_sum = 0
    while(block_sum < size):
      bs = np.random.choice(np.arange(min_block_size,max_block_size+1), size=1)[0]
      if (bs + block_sum <= size):
        noise_block = (np.random.uniform(low = 0.0, high = 0.3, size = (bs,bs))).astype(float)
        mat[block_sum:block_sum + bs , block_sum : block_sum + bs, i] += noise_block
        block_sum += bs
      else :
        bs = size - block_sum
        noise_block = (np.random.uniform(low = 0.0, high = 0.3, size = (bs,bs))).astype(float)
        mat[block_sum:block_sum + bs , block_sum : block_sum + bs, i] += noise_block
        block_sum += bs
  return mat

def main_block_gen(mat):
  size = np.shape(mat)[0]
  n = np.shape(mat)[-1]
  vec_out = np.zeros((size,n))
  for j in range(0,n):
    block_start_vec = np.sort(np.random.choice(np.arange(1,size), size= int(size*0.1)))
    vec_out[0,j] = 1
    vec_out[block_start_vec,j] = 1
    bs = block_start_vec[0]
    diag_block = np.random.uniform(low = 0.5, high = 0.7, size = (bs,bs))
    mat[0:bs,0:bs,j] += diag_block
    for i in range(0,len(block_start_vec)-1):
      bs = block_start_vec[i+1] - block_start_vec[i]
      diag_block = np.random.uniform(low = 0.5, high = 0.7, size = (bs,bs))
      mat[block_start_vec[i]: block_start_vec[i] + bs,block_start_vec[i]: block_start_vec[i] + bs,j] += diag_block
    bs = size - block_start_vec[-1]
    diag_block = np.random.uniform(low = 0.5, high = 0.7, size = (bs,bs))
    mat[block_start_vec[-1]: size,block_start_vec[-1]: size,j] += diag_block
  mat = np.array(mat)
  return mat,vec_out

vec_replacer = np.vectorize(non_zero_replacer)

def create_samples_trial(dim, n, non_zero_ratio = 0.4):
  sample_matrix = (np.random.choice([0.0,1.0], size=(dim,dim,n), p=[1-non_zero_ratio,non_zero_ratio])).astype(float)#np.random.uniform(low=0.0, high=1.0, size=(dim,dim,n))
  sample_matrix = vec_replacer(sample_matrix)
  sample_matrix = block_noise_adder(5,15,sample_matrix)
  sample_matrix, sample_vector = main_block_gen(sample_matrix)
  return sample_matrix,sample_vector

import time

def endindices(x):
  y=np.where(x == 1)[0]
  y = np.append(y,128)
  return y
def Make_Ddash_inverse(D,endindices):
  Ddash_inverse=np.zeros((128,128))
  outtime = 0
  for i in range(1,len(endindices)):
    dcur=D[endindices[i-1]:endindices[i],endindices[i-1]:endindices[i]]
    star = time.time()
    dcurinv=np.linalg.inv(dcur)
    stop = time.time()
    outtime = max(stop-star,outtime)
    Ddash_inverse[endindices[i-1]:endindices[i],endindices[i-1]:endindices[i]]=dcurinv
  return Ddash_inverse,outtime

class gmres_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

from scipy.sparse import csc_matrix


from tqdm import tqdm
import time
import scipy
def time_gain(y,x):

  n = np.shape(y)[-1]
  tg = []
  mean_time = 0
  vanilla_iter = []
  pre_con_iter = []
  for i in tqdm(range(0,n)):
    b = np.array([1.0]*128)
    start = time.time()
    counter1 = gmres_counter()
    
    z1 = scipy.sparse.linalg.gmres(y[:,:,i], b, x0=None, tol=1e-05, restart=200, maxiter=None, M=None,callback=counter1)#jax.scipy.sparse.linalg.gmres(y[:,:,i], b, x0=None,tol=1e-02, atol=1.0, restart=20, maxiter=None, M=None, solve_method='batched')#scipy.sparse.linalg.gmres(y[:,:,i], b, x0=None, tol=1e-02, atol=1, restart=20, maxiter=None, M=None,callback=counter1)
    vanilla_iter.append(counter1.niter)
    end = time.time()
    st1 = time.time()
    j=endindices(x[:,i])
    e1 = time.time()
    X,ttg=Make_Ddash_inverse(y[:,:,i],j)
    ttg = -1*ttg
    ttg += end - start
    mean_time += (end-start)
    start = time.time()
    mat_u = X@ y[:,:,i]
    mat_b = X@b
    
    #counter = gmres_counter()
    #counter = gmres_counter()
    counter2 = gmres_counter()
    z2 = scipy.sparse.linalg.gmres(mat_u, mat_b, x0=None, tol=1e-05, restart=200, maxiter=None, M=None,callback=counter2)#jax.scipy.sparse.linalg.gmres(mat_u, mat_b , x0=None, tol=1e-02, atol=1.0, restart=20, maxiter=None, M=None, solve_method='batched')#scipy.sparse.linalg.gmres(mat_u, mat_b, x0=None, tol=1e-02, atol=1.0, restart=20, maxiter=None, M=None,callback=counter2)
    pre_con_iter.append(counter2.niter)
    #print(counter2.niter)
    end = time.time()
    ttg -= end - start
    tg.append(ttg)
  print(mean_time/(n))
  return tg,vanilla_iter,pre_con_iter

def extract_diagonals(test_data):
    # load the data and move the sample axis to the front
    data = np.moveaxis(np.array(test_data), -1, 0)

    width = data.shape[1]
    samples = data.shape[0]
    diagonalset = np.zeros((samples, 2 * WINDOW + 1, width), dtype=np.float32)

    for j in range(samples):
        image = data[j, :, :]
        # always reallocate the diagonals image here to fill the left triangle with ones
        out = np.ones((2 * WINDOW + 1, width), dtype=np.float32)
        for i in range(-WINDOW, WINDOW + 1):
            diagonal = np.diagonal(image, i)
            out[i + WINDOW, abs(i):] = diagonal
        out = ((out - out.min()) / out.max() * 2) - 1
        diagonalset[j] = out
    return diagonalset
    # remove the previous diago

def prediction_to_vec(prediction):
  out = np.zeros(np.shape(prediction))
  out[prediction>=0.5] = 1
  return out
