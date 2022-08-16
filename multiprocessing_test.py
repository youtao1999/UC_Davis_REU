from multiprocessing import Pool
import time
import numpy as np
import random
from tfim_lanczos import lanczos
import os

num_iter = 100
seed_range = [random.randrange(1, 1e3, 1) for i in range(num_iter)]
h_x_range = np.concatenate((np.linspace(0.1, 4., 50), np.linspace(4.5, 10, 5)))
PBC = True
h_z = 0.001
maxiter = 400
L = [4,4]

def lanczos_single_var(seed):
    return lanczos(L, seed, h_x_range, PBC, h_z, maxiter)

if __name__ == '__main__':
    init = time.time()
    p = Pool()

    exc_eigenvalues_all = np.zeros((len(seed_range), len(h_x_range)))
    first_excited__exc_energies_all = np.zeros((len(seed_range), len(h_x_range)))
    second_derivative_exc_eigenvalues_all = np.zeros((len(seed_range), len(h_x_range)))
    chi_arr_all = np.zeros((len(seed_range), len(h_x_range)))
    S_SG_arr_all = np.zeros((len(seed_range), len(h_x_range)))
    EE_arr_all = np.zeros((len(seed_range), len(h_x_range)))

    result_all = p.map(lanczos_single_var, seed_range, chunksize=10)
    print('multiprocessing_time: ', time.time() - init)

    # N, h_x_range, exc_eigenvalues, first_excited__exc_energies, second_derivative_exc_eigenvalues, chi_arr, S_SG_arr, entropy_arr

    for i, seed in enumerate(seed_range):
        N, h_x_range, exc_eigenvalues, first_excited__exc_energies, second_derivative_exc_eigenvalues, chi_arr, S_SG_arr, entropy_arr = result_all[i]
        exc_eigenvalues_all[i] = exc_eigenvalues
        first_excited__exc_energies_all[i] = first_excited__exc_energies
        second_derivative_exc_eigenvalues_all[i] = second_derivative_exc_eigenvalues
        chi_arr_all[i] = chi_arr
        S_SG_arr_all[i] = S_SG_arr
        EE_arr_all[i] = entropy_arr

    # output files
    # check to see whether the output file already exists
    output = "multiprocessing_test_output"
    if os.path.isdir(output):
        os.chdir(output)
    else:
        os.mkdir(output)
        os.chdir(output)

    if os.path.exists('data_all_{size}.npy'.format(size=L)):
        os.remove('data_all_{size}.npy'.format(size=L))

    with open('data_all_{size}.npy'.format(size=L), 'wb') as f:
        # save comprehensive data
        np.save(f, exc_eigenvalues_all)
        np.save(f, first_excited__exc_energies_all)
        np.save(f, chi_arr_all)
        np.save(f, S_SG_arr_all)
        np.save(f, EE_arr_all)

    os.chdir('../')
