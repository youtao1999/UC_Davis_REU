from multiprocessing import Pool
import time
import numpy as np
import random
from tfim_lanczos import chi_ii
import os

num_iter = 100
seed_range = [random.randrange(1, 1e3, 1) for i in range(num_iter)]
# h_x_range = np.array([10.])
h_x_range = np.concatenate((np.linspace(0.1, 4., 50), np.linspace(4.5, 10, 5)))
PBC = True
h_z = 0.001
maxiter = 400
L = [4,4]
N = L[0] * L[1]

def chi_ii_single_var(seed):
    return chi_ii(L, seed, h_x_range, PBC, h_z, maxiter)

if __name__ == '__main__':
    init = time.time()
    p = Pool()
    result_all = p.map(chi_ii_single_var, seed_range, chunksize=10)
    print('multiprocessing_time: ', time.time() - init)
    chi_ii_arr = np.zeros((len(h_x_range), N, num_iter))
    for i, seed in enumerate(seed_range):
        N, h_x_range, chi_ii_matrix = result_all[i]
        # print(chi_ii_matrix)
        for j in range(N):
            for l in range(num_iter):
                for k in range(len(h_x_range)):
                    chi_ii_arr[k, j, i] = chi_ii_matrix[k, j]
    chi_ii_arr = chi_ii_arr.reshape((len(h_x_range), N * num_iter))
    print(len(np.argwhere(chi_ii_arr == 0.)))
    # output files
    # check to see whether the output file already exists
    output = "chi_ii_prob"
    if os.path.isdir(output):
        os.chdir(output)
    else:
        os.mkdir(output)
        os.chdir(output)

    file_name = '{size}.npy'.format(size=L)
    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, 'wb') as f:
        # save comprehensive data
        np.save(f, chi_ii_arr)

    os.chdir('../')
