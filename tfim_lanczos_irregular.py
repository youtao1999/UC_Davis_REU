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
h_x_range = np.concatenate((np.linspace(init, final, num_steps), np.linspace(final+0.5, final+6., 5)))
h_z = 0.001
maxiter = 400

N = 8
partition_set_8 = [[[0,2,3,6], [1,4,5,7]], [[0,2,5,7], [1,3,4,6]], [[2,3,6,7],[0,1,4,5]], [[2,6,1,5], [0,3,4,7]]]
partition_set_10 = [[[0,1,3,4,7], [2,5,6,8,9]], [[1,2,4,5,8], [0,3,7,6,9]], [[0,1,2,3,4], [5,6,7,8,9]], [[0,1,2,8,9], [3,4,5,6,7]]]

# compute entanglement entropy
def partition_basis(GS_indices, A, B):

    # building initial basis as set since to avoid repeated elements
    A_basis = set()
    B_basis = set()

    for index, GS_index in np.ndenumerate(GS_indices):
        state = tfim_lanczos.state(GS_index, N)
        A_basis.add(tuple(state[A]))
        B_basis.add(tuple(state[B]))

    def sum_digits(digits):
        return sum(c << i for i, c in enumerate(digits))

        # now we extract the elements from this set and start building the ordered

    # basis

    # reordering basis A
    index_matching_A = {}

    for ele in A_basis:
        ele = np.array(ele)
        index_matching_A[sum_digits(ele)] = np.array(ele)

    A_reordered_basis = np.zeros((len(index_matching_A), len(list(index_matching_A.values())[0])))
    for index, key in enumerate(index_matching_A.keys()):
        A_reordered_basis[index] = index_matching_A[key]

    # reordering basis B
    index_matching_B = {}

    for ele in B_basis:
        ele = np.array(ele)
        index_matching_B[sum_digits(ele)] = np.array(ele)

    B_reordered_basis = np.zeros((len(index_matching_B), len(list(index_matching_B.values())[0])))
    for index, key in enumerate(index_matching_B.keys()):
        B_reordered_basis[index] = index_matching_B[key]

    return A_reordered_basis, B_reordered_basis

def beta_ij(GS_indices, A, B, overall_GS, perturbation_param_index):
    # this function builds beta_ij matrix as a function of the index of the perturbation parameter

    # build A, B basis first
    A_basis, B_basis = partition_basis(GS_indices, A, B)
    s = overall_GS[perturbation_param_index]

    def find(basis, target):
        # this function finds the index of the target state in the basis state, gives the index of 1 in BETA
        for i, state in enumerate(basis):
            if np.array_equiv(state, target):
                return i

    BETA = np.zeros((len(A_basis), len(B_basis)))

    for (probability, GS_index) in zip(s, GS_indices):
        GS_state = tfim_lanczos.state(GS_index, N)
        i = find(A_basis, GS_state[A])
        j = find(B_basis, GS_state[B])
        BETA[i, j] += probability
    return BETA

def lanczos_entropy(GS_indices, A, B, overall_GS, perturbation_param_index):

    BETA = beta_ij(GS_indices, A, B, overall_GS, perturbation_param_index)

    # perform uv decomposition

    u, s, vh = np.linalg.svd(BETA, full_matrices=True)

    s = s[np.where(s != 0.)]
    # add conditional statement to remove all the zero singular values

    entropy = -np.dot(s ** 2, np.log(s ** 2))

    return entropy

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

def lanczos_irregular(shape, seed, h_x_range, h_z, maxiter):
    start_time = time.time()
    if shape == 8:
        Jij_func = tfim_perturbation.eight_tile
        partition_set = partition_set_8
    elif shape == 10:
        Jij_func = tfim_perturbation.ten_tile
        partition_set = partition_set_10

    Jij, N = Jij_func(seed, p=0.5)

    Ising_energy_arr = Ising_energies(Jij)
    GS_energy, GS_indices = tfim_perturbation.GS(Ising_energy_arr)

    # initialize Lanczos vector
    v0 = np.zeros(2 ** N)
    for i in GS_indices:
        v0[i] = 1

    # Calculate exact eigenvalues and eigenstates for range(h_x)
    V_exc_csr, H_0_exc_csr, exc_eigenvalues, first_excited__exc_energies, exc_eigenstates = tfim_lanczos.exc_eigensystem(
        h_x_range, Ising_energy_arr, N, v0)

    print(
        "----{num_sec}s seconds ---- used for diagonalization for seed {seed}".format(num_sec=time.time() - start_time,
                                                                                      seed=seed))

    first_derivative_exc_eigenvalues = np.gradient(exc_eigenvalues, (final - init) / float(num_steps))
    second_derivative_exc_eigenvalues = np.gradient(first_derivative_exc_eigenvalues, (final - init) / float(num_steps))

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
            #         H = - V_exc.multiply(h_x) - h_z*sparse.diags(sigma_z)
            longitudinal_energy = \
                spla.eigsh(H_0_exc_csr - V_exc_csr.multiply(h_x) - h_z * sparse.diags(sigma_z), k=1, which='SA', v0=v0,
                           maxiter=maxiter,
                           return_eigenvectors=False)[0]
            chi_aa = 2. * abs(abs(exc_eigenvalues[i]) - abs(longitudinal_energy)) / (h_z ** 2)
            chi_aa_matrix[i, a] += chi_aa

    # In[18]:
    chi_ab_matrix = np.zeros((len(h_x_range), N, N))
    for n, h_x in enumerate(h_x_range):
        for a in range(N):
            sigma_z_a = np.zeros(2 ** N)
            for ket in range(2 ** N):
                state_1 = tfim_lanczos.state(ket, N)
                if state_1[a] == 1:
                    sigma_z_a[ket] += 1
                else:
                    sigma_z_a[ket] -= 1
            for b in range(a + 1, N, 1):
                sigma_z_b = np.zeros(2 ** N)
                for ket in range(2 ** N):
                    state_2 = tfim_lanczos.state(ket, N)
                    if state_2[b] == 1:
                        sigma_z_b[ket] += 1
                    else:
                        sigma_z_b[ket] -= 1
                H = H_0_exc_csr - V_exc_csr.multiply(h_x) - (
                            sparse.diags(sigma_z_a) + sparse.diags(sigma_z_b)).multiply(
                    h_z)
                longitudinal_energy = spla.eigsh(H, k=1, which='SA', v0=v0, maxiter=maxiter, return_eigenvectors=False)[
                    0]
                chi_ab = (exc_eigenvalues[n] - longitudinal_energy) / (h_z ** 2.) - 0.5 * (
                        chi_aa_matrix[n, a] + chi_aa_matrix[n, b])
                chi_ab_matrix[n, a, b] += chi_ab
                chi_ab_matrix[n, b, a] += chi_ab
                # adding the diagonal elements
                for c in range(N):
                    chi_ab_matrix[n, c, c] = chi_aa_matrix[n, c]
    chi_arr = np.zeros(len(h_x_range))
    for k, h_x in enumerate(h_x_range):
        chi_arr[k] += np.sum(np.power(chi_ab_matrix[k],2.))
    print("----{num_sec}s seconds ---- used for susceptibility for seed {seed}".format(
        num_sec=time.time() - susceptibility_time, seed=seed))

    structure_factor_time = time.time()
    # compute structure factor
    S_SG_arr = np.zeros(np.shape(h_x_range))
    for m, h_x in enumerate(h_x_range):
        psi0 = exc_eigenstates[m]
        for a in range(N):
            for b in range(N):
                sigma_z_a = np.zeros(2 ** N)
                sigma_z_b = np.zeros(2 ** N)
                for ket in range(2 ** N):
                    state_1 = tfim_lanczos.state(ket, N)
                    if state_1[a] == 1:
                        sigma_z_a[ket] += 1
                    else:
                        sigma_z_a[ket] -= 1
                for ket in range(2 ** N):
                    state_2 = tfim_lanczos.state(ket, N)
                    if state_2[b] == 1:
                        sigma_z_b[ket] += 1
                    else:
                        sigma_z_b[ket] -= 1
                S_ab = psi0 @ sparse.diags(sigma_z_a) @ sparse.diags(sigma_z_b) @ psi0.T
                S_SG_arr[m] += S_ab**2.
    print("----{num_sec}s seconds ---- used for structure factor for seed {seed}".format(
        num_sec=time.time() - structure_factor_time, seed=seed))

    # calculate entanglement entropy
    lanczos_entropy_par_arr = np.zeros((len(partition_set), len(h_x_range)))
    for k, par in enumerate(partition_set):
        [A, B] = par
        for j in range(len(h_x_range)):
            lanczos_entropy_par_arr[k, j] = lanczos_entropy([i for i in range(2 ** N)], A, B, exc_eigenstates, j)
    lanczos_entropy_arr = np.mean(lanczos_entropy_par_arr, axis=0)

    return shape, h_x_range, exc_eigenvalues, first_excited__exc_energies, chi_arr, S_SG_arr, lanczos_entropy_arr

def lanczos_irregular_single_var(seed):
    return lanczos_irregular(N, seed, h_x_range, h_z, maxiter)

if __name__ == '__main__':

    init = time.time()
    p = Pool()

    exc_eigenvalues_all = np.zeros((num_iter, len(h_x_range)))
    first_excited__exc_energies_all = np.zeros((num_iter, len(h_x_range)))
    chi_arr_all = np.zeros((num_iter, len(h_x_range)))
    S_SG_arr_all= np.zeros((num_iter, len(h_x_range)))
    EE_arr_all= np.zeros((num_iter, len(h_x_range)))

    result_all = p.map(lanczos_irregular_single_var, seed_list, chunksize=10)
    print('multiprocessing_time: ', time.time() - init)

    for seed_index, seed in enumerate(seed_list):
        shape, h_x_range, exc_eigenvalues, first_excited__exc_energies, chi_arr, S_SG_arr, lanczos_entropy_arr = result_all[seed_index]
        exc_eigenvalues_all[seed_index] = exc_eigenvalues
        first_excited__exc_energies_all[seed_index] = first_excited__exc_energies
        chi_arr_all[seed_index] = chi_arr
        S_SG_arr_all[seed_index] = S_SG_arr
        EE_arr_all[seed_index] = lanczos_entropy_arr

    # output files
    # check to see whether the output file already exists
    output = "multiprocessing_test_output"
    if os.path.isdir(output):
        os.chdir(output)
    else:
        os.mkdir(output)
        os.chdir(output)
    if os.path.exists('data_all_{size}.npy'.format(size = shape)):
        os.remove('data_all_{size}.npy'.format(size = shape))

    with open('data_all_{size}.npy'.format(size = shape), 'wb') as f:
        # save comprehensive data
        np.save(f, exc_eigenvalues_all)
        np.save(f, first_excited__exc_energies_all)
        np.save(f, chi_arr_all)
        np.save(f, S_SG_arr_all)
        np.save(f, EE_arr_all)

    os.chdir('../')
