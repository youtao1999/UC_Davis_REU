#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tfim
import tfim_perturbation
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
import time
import tfim_EE
import tfim_rdm

# functionalize the diagonalization and data production procedure
def lanczos(L, seed, h_x_range, PBC, h_z, maxiter):

    # In[3]:

    start_time = time.time()
    def Jij_2D_NN(seed, N, PBC, xwidth, yheight, lattice, p):

        def bond_list_unequal(seed, N, PBC, xwidth, yheight, p):
            # p is the probability distribution of ferromagnetic bonds
            np.random.seed(seed)
            if PBC == True:
                num_of_bonds = 2*N
            else:
                num_of_bonds = (xwidth - 1)*(yheight) + (xwidth)*(yheight - 1)
            i = [np.random.random() for _ in range(num_of_bonds)]
            # print(i)
            a = np.zeros(len(i))
            for index, prob_seed in enumerate(i):
                if prob_seed <= p:
                    a[index] += 1
                else:
                    a[index] -= 1
            return a

        def make_Jij(N, b_list, lattice):
            #Goes through the list of bonds to make the jij matrix that tells you how all of the spins are bonded to each other

            bond_index = 0
            Jij = np.zeros((N,N))
            for i in range(0,N):
                NNs = lattice.NN(i)
                for j in NNs:
                    if Jij[i][j] == 0:
                        Jij[i][j] = b_list[bond_index]
                        Jij[j][i] = b_list[bond_index]
                        bond_index += 1
            return Jij

        b_list = bond_list_unequal(seed, N, PBC, xwidth, yheight, p)
        return make_Jij(N, b_list, lattice)


    # In[4]:


    # Build lattice and basis
    lattice = tfim.Lattice(L, PBC)
    N = lattice.N
    basis = tfim.IsingBasis(lattice)


    # In[5]:


    #construct random J matrix
    Jij = Jij_2D_NN(seed, N, PBC, L[0], L[1], lattice, p = 0.5)


    # In[6]:


    # List out all the spin_states, corresponding indices and energies
    Ising_energy_arr = np.zeros(2**N)
    for index in range(2**N):
        state = basis.state(index)
        # modify state from 0 and 1 base to -1, 1 base
        for i in range(N):
            if state[i] == 0:
                state[i] -= 1
        Ising_energy = 0
        for i in range(N):
            for j in range(i+1, N, 1):
                bond_energy = Jij[i, j] * state[i] * state[j]
                Ising_energy += bond_energy
        Ising_energy_arr[index] = Ising_energy

    # In[7]:


    GS_energy, GS_indices = tfim_perturbation.GS(Ising_energy_arr)

    # initialize Lanczos vector
    v0 = np.zeros(2**N)
    for i in GS_indices:
        v0[i] = 1


    # In[8]:


    # modified exact Hamiltonians using compressed sparse row matrices
    def V_exact_csr(basis, lattice):
        row = []
        col = []
        for ket in range(basis.M):
            state = basis.state(ket)
            for i in range(lattice.N):
                basis.flip(state,i)
                bra = basis.index(state)
                row.append(bra)
                col.append(ket)
                basis.flip(state,i)
        data = np.ones(len(col))
        V_exact = sparse.csr_matrix((data, (np.array(row), np.array(col))), shape = (2**N, 2**N))
        return V_exact

    def H_0_exact_csr(Energies):
        return sparse.diags(Energies)

    # modified function to eigendecompose the exact Hamiltonian using Lanczos method
    def exc_eigensystem(basis, h_x_range, lattice, Energies):
        # Calculate exact eigenvalues and eigenstates for range(h_x)
        exc_eigenvalues = np.zeros(len(h_x_range))
        first_excited_exc_energies = np.zeros(len(h_x_range))
        exc_eigenstates = np.zeros((len(h_x_range), basis.M))
        V_exc_csr = V_exact_csr(basis, lattice)
        H_0_exc_csr = H_0_exact_csr(Energies)
        for j, h_x in enumerate(h_x_range):
            H = H_0_exc_csr - V_exc_csr.multiply(h_x)
            exc_eigenvalue, exc_eigenstate = spla.eigsh(H, k = 2, which = 'SA', v0 = v0, maxiter = maxiter, tol = 1e-5, return_eigenvectors = True)

            exc_eigenvalues[j] = exc_eigenvalue[0]
            first_excited_exc_energies[j] = exc_eigenvalue[1]
            for k in range(basis.M):
                exc_eigenstates[j][k] = exc_eigenstate[k, 0]
        return V_exc_csr, H_0_exc_csr, exc_eigenvalues, first_excited_exc_energies, exc_eigenstates


    # In[10]:


    # Calculate exact eigenvalues and eigenstates for range(h_x)
    V_exc, H_0_exc, exc_eigenvalues, first_excited__exc_energies, exc_eigenstates = exc_eigensystem(basis, h_x_range, lattice, Ising_energy_arr)

    # print("----%s seconds ----" % (time.time() - start_time))

    # In[13]:

    final = h_x_range[-1]
    init = h_x_range[1]
    num_steps = len(h_x_range)
    # first and second derivative of ground state energy per site
    first_derivative_exc_eigenvalues = np.gradient(exc_eigenvalues, (final-init)/float(num_steps))
    second_derivative_exc_eigenvalues = np.gradient(first_derivative_exc_eigenvalues, (final-init)/float(num_steps))


    # compute susciptibility

    # In[16]:


    chi_aa_matrix = np.zeros((len(h_x_range), lattice.N))
    for i, h_x in enumerate(h_x_range):
        for a in range(lattice.N):
            sigma_z = np.zeros(basis.M)
            for ket in range(basis.M):
                state = basis.state(ket)
                if state[a] == 1:
                    sigma_z[ket] += 1
                else:
                    sigma_z[ket] -= 1
            longitudinal_energy = spla.eigsh(H_0_exc - V_exc.multiply(h_x) - h_z*sparse.diags(sigma_z), k = 1, which = 'SA', v0 = v0, tol = 1e-5, maxiter = maxiter, return_eigenvectors = False)[0]
            # print("----%s seconds for h_x = %s----" % (time.time() - start_time, h_x))
            chi_aa = 2.*(exc_eigenvalues[i] - longitudinal_energy)/(h_z**2)
            chi_aa_matrix[i, a] += chi_aa


    # In[18]:


    chi_ab_matrix = np.zeros((len(h_x_range), basis.N, basis.N))
    for i, h_x in enumerate(h_x_range):
        for a in range(lattice.N):
            sigma_z_a = np.zeros(basis.M)
            for ket in range(basis.M):
                state = basis.state(ket)
                if state[a] == 1:
                    sigma_z_a[ket] += 1
                else:
                    sigma_z_a[ket] -= 1
            for b in range(a+1, lattice.N, 1):
                sigma_z_b = np.zeros(basis.M)
                for ket in range(basis.M):
                    state = basis.state(ket)
                    if state[b] == 1:
                        sigma_z_b[ket] += 1
                    else:
                        sigma_z_b[ket] -= 1
                H = H_0_exc - V_exc.multiply(h_x) - (sparse.diags(sigma_z_a) + sparse.diags(sigma_z_b)).multiply(h_z)
                longitudinal_energy = spla.eigsh(H, k = 1, which = 'SA', v0 = v0, tol = 1e-5, maxiter = maxiter, return_eigenvectors = False)[0]
                # print("----%s seconds for h_x = %s----" % (time.time() - start_time, h_x))
                chi_ab = (exc_eigenvalues[i]-longitudinal_energy)/(h_z**2.) - 0.5*(chi_aa_matrix[i, a] + chi_aa_matrix[i, b])
                chi_ab_matrix[i, a, b] += chi_ab
                chi_ab_matrix[i, b, a] += chi_ab
            # adding the diagonal elements
            for c in range(N):
                chi_ab_matrix[i, c, c] = chi_aa_matrix[i, c]

    chi_arr = np.zeros(len(h_x_range))
    for i, h_x in enumerate(h_x_range):
        chi_arr[i] += np.sum(np.power(chi_ab_matrix[i],2))
    # print("----%s seconds ----" % (time.time() - start_time))

    # compute structure factor
    S_SG_arr = np.zeros(np.shape(h_x_range))
    for i, h_x in enumerate(h_x_range):
        psi0 = exc_eigenstates[i]
        for a in range(N):
            for b in range(N):
                sigma_z_a = np.zeros(basis.M)
                sigma_z_b = np.zeros(basis.M)
                for ket in range(basis.M):
                    state = basis.state(ket)
                    if state[a] == 1:
                        sigma_z_a[ket] += 1
                    else:
                        sigma_z_a[ket] -= 1
                for ket in range(basis.M):
                    state = basis.state(ket)
                    if state[b] == 1:
                        sigma_z_b[ket] += 1
                    else:
                        sigma_z_b[ket] -= 1
                S_ab = psi0 @ sparse.diags(sigma_z_a) @ sparse.diags(sigma_z_b) @ psi0.T
                S_SG_arr[i] += S_ab**2.
    # print("----%s seconds ----" % (time.time() - start_time))

    # compute entanglement entropy
    partition_set = tfim_EE.linear_bipartition(L)
    entropy_par_arr = np.zeros((len(partition_set), len(h_x_range)))
    #for par in partition
    for k in range(len(partition_set)):
        [A, B] = partition_set[k]
        for i, h_x in enumerate(h_x_range):
            psi0 = exc_eigenstates[i]
            S, U, V = tfim_rdm.svd(basis, A, B, psi0)
            entropy = tfim_rdm.entropy(S)
            entropy_par_arr[k, i] = entropy
    entropy_par_ave = np.mean(entropy_par_arr, axis = 0)

    print('for seed ', seed, 'time used ', time.time() - start_time)
    return N, h_x_range, exc_eigenvalues, first_excited__exc_energies, second_derivative_exc_eigenvalues, chi_arr, S_SG_arr, entropy_par_ave

def chi_ii(L, seed, h_x_range, PBC, h_z, maxiter):

    # In[3]:

    start_time = time.time()
    def Jij_2D_NN(seed, N, PBC, xwidth, yheight, lattice, p):

        def bond_list_unequal(seed, N, PBC, xwidth, yheight, p):
            # p is the probability distribution of ferromagnetic bonds
            np.random.seed(seed)
            if PBC == True:
                num_of_bonds = 2*N
            else:
                num_of_bonds = (xwidth - 1)*(yheight) + (xwidth)*(yheight - 1)
            i = [np.random.random() for _ in range(num_of_bonds)]
            # print(i)
            a = np.zeros(len(i))
            for index, prob_seed in enumerate(i):
                if prob_seed <= p:
                    a[index] += 1
                else:
                    a[index] -= 1
            return a

        def make_Jij(N, b_list, lattice):
            #Goes through the list of bonds to make the jij matrix that tells you how all of the spins are bonded to each other

            bond_index = 0
            Jij = np.zeros((N,N))
            for i in range(0,N):
                NNs = lattice.NN(i)
                for j in NNs:
                    if Jij[i][j] == 0:
                        Jij[i][j] = b_list[bond_index]
                        Jij[j][i] = b_list[bond_index]
                        bond_index += 1
            return Jij

        b_list = bond_list_unequal(seed, N, PBC, xwidth, yheight, p)
        return make_Jij(N, b_list, lattice)


    # In[4]:


    # Build lattice and basis
    lattice = tfim.Lattice(L, PBC)
    N = lattice.N
    basis = tfim.IsingBasis(lattice)


    # In[5]:


    #construct random J matrix
    Jij = Jij_2D_NN(seed, N, PBC, L[0], L[1], lattice, p = 0.5)


    # In[6]:


    # List out all the spin_states, corresponding indices and energies
    Ising_energy_arr = np.zeros(2**N)
    for index in range(2**N):
        state = basis.state(index)
        # modify state from 0 and 1 base to -1, 1 base
        for i in range(N):
            if state[i] == 0:
                state[i] -= 1
        Ising_energy = 0
        for i in range(N):
            for j in range(i+1, N, 1):
                bond_energy = Jij[i, j] * state[i] * state[j]
                Ising_energy += bond_energy
        Ising_energy_arr[index] = Ising_energy

    # In[7]:


    GS_energy, GS_indices = tfim_perturbation.GS(Ising_energy_arr)

    # initialize Lanczos vector
    v0 = np.zeros(2**N)
    for i in GS_indices:
        v0[i] = 1


    # In[8]:


    # modified exact Hamiltonians using compressed sparse row matrices
    def V_exact_csr(basis, lattice):
        row = []
        col = []
        for ket in range(basis.M):
            state = basis.state(ket)
            for i in range(lattice.N):
                basis.flip(state,i)
                bra = basis.index(state)
                row.append(bra)
                col.append(ket)
                basis.flip(state,i)
        data = np.ones(len(col))
        V_exact = sparse.csr_matrix((data, (np.array(row), np.array(col))), shape = (2**N, 2**N))
        return V_exact

    def H_0_exact_csr(Energies):
        return sparse.diags(Energies)

    V_exc_csr = V_exact_csr(basis, lattice)
    H_0_exc_csr = H_0_exact_csr(Ising_energy_arr)

    chi_aa_matrix = np.zeros((len(h_x_range), lattice.N))
    for i, h_x in enumerate(h_x_range):
        for a in range(lattice.N):
            sigma_z = np.zeros(basis.M)
            for ket in range(basis.M):
                state = basis.state(ket)
                if state[a] == 1:
                    sigma_z[ket] += 1
                else:
                    sigma_z[ket] -= 1
            longitudinal_energy = spla.eigsh(H_0_exc_csr - V_exc_csr.multiply(h_x) - h_z*sparse.diags(sigma_z), k = 1, which = 'SA', v0 = v0, tol = 1e-5, maxiter = maxiter, return_eigenvectors = False)[0]
            exc_eigenvalue  = spla.eigsh(H_0_exc_csr - V_exc_csr.multiply(h_x), k=1, which='SA', v0=v0,
                       tol=1e-5, maxiter=maxiter, return_eigenvectors=False)[0]
            chi_aa = 2.*(exc_eigenvalue - longitudinal_energy)/(h_z**2)
            chi_aa_matrix[i, a] += chi_aa
    print("----%s seconds for seed %s----" % (time.time() - start_time, seed))
    return N, h_x_range, chi_aa_matrix

# modified exact Hamiltonians using compressed sparse row matrices
def flip(state, i):
    """Flips ith spin in state"""
    state[i] = (state[i] + 1) % 2

def state(index, N):
    """Returns the state associated with index"""
    return np.array(list(bin(index)[2:].zfill(N))).astype(int)

def stateindex(state):
    """Returns the index associated with state"""
    return int(''.join(state.astype(str)),2)

def V_exact_csr(N):
    row = []
    col = []

    for ket in range(2**N):
        state_1 = state(ket, N)
        for i in range(N):
            flip(state_1,i)
            bra = stateindex(state_1)
            row.append(bra)
            col.append(ket)
            flip(state_1,i)
    data = np.ones(len(col))
    V_exact = sparse.csr_matrix((data, (np.array(row), np.array(col))), shape = (2**N, 2**N))
    return V_exact

def H_0_exact_csr(Energies):
    return sparse.diags(Energies)

# modified function to eigendecompose the exact Hamiltonian using Lanczos method
def exc_eigensystem(h_x_range, Energies, N, v0):
    # Calculate exact eigenvalues and eigenstates for range(h_x)
    exc_eigenvalues = np.zeros(len(h_x_range))
    first_excited_exc_energies = np.zeros(len(h_x_range))
    exc_eigenstates = np.zeros((len(h_x_range), 2**N))
    V_exc_csr = V_exact_csr(N)
    H_0_exc_csr = H_0_exact_csr(Energies)
    for j, h_x in enumerate(h_x_range):
        H = H_0_exc_csr - V_exc_csr.multiply(h_x)
        exc_eigenvalue, exc_eigenstate = spla.eigsh(H, k = 4, v0 = v0, maxiter = 200, return_eigenvectors = True)
        exc_eigenvalues[j] = exc_eigenvalue[0]
        first_excited_exc_energies[j] = exc_eigenvalue[1]
        for k in range(2**N):
            exc_eigenstates[j][k] = exc_eigenstate[k, 0]
    return V_exc_csr, H_0_exc_csr, exc_eigenvalues, first_excited_exc_energies, exc_eigenstates



