import tfim_lanczos
import random
import time
import tfim_perturbation
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
import os
from multiprocessing import Pool

num_iter = 100
seed_list = [random.randrange(1, 1e3, 1) for i in range(num_iter)]
init = 0.1
final = 4.
num_steps = 50
# h_x_range = np.array([0.01])
h_x_range = np.concatenate((np.linspace(init, final, num_steps), np.linspace(final+0.5, final+6., 5)))
h_z = 0.001
maxiter = 400

N = 10

def Ising_energies(Jij):

    Ising_energy_arr = np.zeros(2 ** N)
    for index in range(2 ** N):
        state_1 = tfim_lanczos.state(index, N)
        # modify state from 0 and 1 base to -1, 1 base
        for i in range(N):
            if state_1[i] == 0:
                state_1[i] -= 1
        Ising_energy = 0
        for i in range(N):
            for j in range(i + 1, N, 1):
                bond_energy = Jij[i, j] * state_1[i] * state_1[j]
                Ising_energy += bond_energy
        Ising_energy_arr[index] = Ising_energy

    return Ising_energy_arr

def chi_ii_irregular(N, seed, h_x_range, h_z, maxiter):

    if N == 8:
        Jij_func = tfim_perturbation.eight_tile
    elif N == 10:
        Jij_func = tfim_perturbation.ten_tile

    Jij, N = Jij_func(seed, p=0.5)

    Ising_energy_arr = Ising_energies(Jij)
    GS_energy, GS_indices = tfim_perturbation.GS(Ising_energy_arr)

    # initialize Lanczos vector
    v0 = np.zeros(2 ** N)
    for i in GS_indices:
        v0[i] = 1

    # Calculate exact eigenvalues and eigenstates for range(h_x)
    V_exc_csr = tfim_lanczos.V_exact_csr(N)
    H_0_exc_csr = tfim_lanczos.H_0_exact_csr(Ising_energy_arr)

    susceptibility_time = time.time()
    chi_aa_matrix = np.zeros((len(h_x_range), N))
    for i, h_x in enumerate(h_x_range):
        for a in range(N):
            sigma_z = np.zeros(2 ** N)
            for ket in range(2 ** N):
                state_1 = tfim_lanczos.state(ket, N)
                if state_1[a] == 1:
                    sigma_z[ket] += 1
                else:
                    sigma_z[ket] -= 1
            exc_eigenvalue = spla.eigsh(H_0_exc_csr - V_exc_csr.multiply(h_x), k=1, which='SA', v0=v0,
                           maxiter=maxiter,
                           return_eigenvectors=False)[0]
            longitudinal_energy = \
                spla.eigsh(H_0_exc_csr - V_exc_csr.multiply(h_x) - h_z * sparse.diags(sigma_z), k=1, which='SA', v0=v0,
                           maxiter=maxiter,
                           return_eigenvectors=False)[0]
            chi_aa = 2. * (exc_eigenvalue - longitudinal_energy) / (h_z ** 2)
            chi_aa_matrix[i, a] += chi_aa

    print("----{num_sec}s seconds ---- used for susceptibility for seed {seed}".format(
        num_sec=time.time() - susceptibility_time, seed=seed))
    return N, h_x_range, chi_aa_matrix

def chi_ii_irregular_single_var(seed):
    return chi_ii_irregular(N, seed, h_x_range, h_z, maxiter)

if __name__ == '__main__':
    init = time.time()
    p = Pool()
    result_all = p.map(chi_ii_irregular_single_var, seed_list, chunksize=10)
    print('multiprocessing_time: ', time.time() - init)
    chi_ii_arr = np.zeros((len(h_x_range), N, num_iter))
    for i, seed in enumerate(seed_list):
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

    file_name = '{size}.npy'.format(size=N)
    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, 'wb') as f:
        # save comprehensive data
        np.save(f, chi_ii_arr)

    os.chdir('../')
