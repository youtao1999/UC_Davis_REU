#!/usr/bin/env python

""""TFIMED.py
    Tao You
    07/02/2020
    --Build first and second order approximated matrices using perturbation theory
    --Requires: numpy, scipy.sparse, scipy.linalg, progressbar
"""
import tfim_matrices as tfim_matrices
import tfim
import numpy as np
from scipy.linalg import eigh
from scipy import sparse
from scipy.sparse import linalg as spla
import networkx as nx


###############################################################################

# To be used in tfim_search:

def Hamming_distance(state_1, state_2):
    # Define Hamming distance
    return len(np.nonzero(state_1 - state_2)[0])

def Hamming_array(GS_indices, basis):
    # Calculate Hamming distance array
    Hamming = np.zeros((len(GS_indices), len(GS_indices)-1))
    for i, n in enumerate(GS_indices):
        basis_n = basis.state(n)
        for j, m in enumerate(np.delete(GS_indices, i)):
            basis_m = basis.state(m)
            Hamming[i, j] = Hamming_distance(basis_n, basis_m)
    return Hamming

def judge(order, array, N):
    if array[np.nonzero(array <= N/2.0)] != []:
        return np.max(array[np.nonzero(array <= N/2.0)]) <= order

# def minOrder(order, array, N):
#     # if this function returns true, then this instance is a threshold instance for which our perturbation expansion is non-trivial
#     if array[np.nonzero(array <= N/2.0)] != []:
#         return np.min(array[np.nonzero(array <= N/2.0)]) == order


def min_order(basis, GS_indices):
    '''
    This function returns the minimum order needed in the perturbative expansion in order to connect the ground manifold.
    Note that by taking advantage of the spin-inversion symmetry, we relax the criterion for the ground manifold being
    connected to maxium two disconnected components which are spin-inversion of each other. We then can artificially
    construct the equal superposition in the wavefunction. Thus the minimum order needed is the one such that the ground
    manifold is reduced to two spin-inversion components of each other.
    '''

    def order_GS_graph(basis, GS_indices, order):
        Hamming_matrix = np.zeros((len(GS_indices), len(GS_indices)))
        for i in range(len(GS_indices)):
            for j in range(i + 1, len(GS_indices)):
                GS_index_1 = GS_indices[i]
                GS_index_2 = GS_indices[j]
                if Hamming_distance(basis.state(GS_index_1), basis.state(GS_index_2)) <= order:
                    Hamming_matrix[i, j] += 1
        Hamming_matrix += Hamming_matrix.T
        return nx.from_numpy_matrix(Hamming_matrix)

    # Construct graph
    order = 0
    G = order_GS_graph(basis, GS_indices, order)
    while nx.number_connected_components(G) > 2:
        order += 1
        G = order_GS_graph(basis, GS_indices, order)

    return order
    
###############################################################################

def state_energy(basis,J,state_index):
    """ Computes specific state energy"""
    
    shift_state = np.zeros(basis.N,dtype=int)
    state = basis.spin_state(state_index)
    energy = 0
    for shift in range(1,int(basis.N/2+1)):
        shift_state[shift:] = state[:-shift]
        shift_state[:shift] = state[-shift:]
        if (basis.N%2 == 0) and (shift == basis.N/2):
            energy = energy + 0.5*np.dot(J[shift-1,:]*shift_state,state)
        else:
            energy = energy + np.dot(J[shift-1,:]*shift_state,state)
    energy = energy*(-1)
    return energy

def GS(Energies):
    GS_energy = np.min(Energies)
    GS_indices = np.nonzero(Energies == GS_energy)[0]
    return GS_energy, GS_indices
    
def H_app_0(GS_energy, GS_indices):
    return GS_energy*np.identity(len(GS_indices))

def H_app_1(basis, GS_indices, N):
    
    # First-Order term in perturbation theory
    V = np.zeros((len(GS_indices), len(GS_indices)))
    
    for column, ket in enumerate(GS_indices):
        state = basis.state(ket)
        for i in range(N):
            basis.flip(state,i)
            bra = basis.index(state)
            subspace_index = np.argwhere(GS_indices == bra)
            if len(subspace_index) > 0:
                row = subspace_index[0][0]
                V[row, column] += 1
            basis.flip(state,i)
    return V

def H_app_2(basis, Jij, GS_indices, N, GS_energy):
    # Second-Order term in perturbation theory
    H_app_2 = np.zeros((len(GS_indices), len(GS_indices)))
    
    for column, GS_ket_1 in enumerate(GS_indices):
        state_1 = basis.state(GS_ket_1)
        for i in range(N):
            basis.flip(state_1, i)
            state_1_flipped_index = basis.index(state_1)
            if state_1_flipped_index not in GS_indices:
                energy_gap = state_energy(basis, Jij, state_1_flipped_index) - GS_energy
                for j in range(N):
                    basis.flip(state_1,j)
                    ES_2_flipped_index = basis.index(state_1)
                    GS_2_index = np.argwhere(np.array(GS_indices) == ES_2_flipped_index)
                    if len(GS_2_index) > 0:
                        row = GS_2_index[0][0]
                        H_app_2[row, column] -= 1./energy_gap
                    basis.flip(state_1, j)
            basis.flip(state_1, i)
    return H_app_2

def H_app_3(basis, Jij, GS_indices, N, GS_energy):
    # 3rd order approximation term
    H_app_3 = np.zeros((len(GS_indices), len(GS_indices)))
    
    for column, GS_bra_1 in enumerate(GS_indices):
        state_0 = basis.state(GS_bra_1)
        for i in range(N):
            basis.flip(state_0, i)
            state_1_index = basis.index(state_0)
            if state_1_index in GS_indices:
                for j in range(N):
                    basis.flip(state_0, j)
                    state_2_index = basis.index(state_0)
                    if state_2_index not in GS_indices:
                        energy_gap = GS_energy - state_energy(basis, Jij, state_2_index)
                        for k in range(N):
                            basis.flip(state_0, k)
                            state_3_index = basis.index(state_0)
                            GS_3_index = np.argwhere(np.array(GS_indices) == state_3_index)
                            if len(GS_3_index) > 0:
                                row = GS_3_index[0][0]
                                H_app_3[row, column] += -0.5/(energy_gap**2)
                            basis.flip(state_0, k)
                    basis.flip(state_0, j)
            basis.flip(state_0, i)
    term_2 = np.transpose(H_app_3)
    H_app_3 += term_2
    
#     for column, GS_bra_1 in enumerate(GS_indices):        
#         state_0 = basis.state(GS_bra_1)
#         for i in range(N):
#             basis.flip(state_0, i)
#             state_1_index = basis.index(state_0)
#             if state_1_index not in GS_indices:
#                 energy_gap = state_energy(basis, Jij, state_1_index) - GS_energy
#                 for j in range(N):
#                     basis.flip(state_0, j)
#                     state_2_index = basis.index(state_0)
#                     if state_2_index in GS_indices:
#                         for k in range(N):
#                             basis.flip(state_0, k)
#                             state_3_index = basis.index(state_0)
#                             GS_3_index = np.argwhere(np.array(GS_indices) == state_3_index)
#                             if len(GS_3_index) > 0:
#                                 row = GS_3_index[0][0]
#                                 H_app_3[row, column] += -0.5/(energy_gap**2)
#                             basis.flip(state_0, k)
#                     basis.flip(state_0, j)
#             basis.flip(state_0, i)

    for column, GS_bra_1 in enumerate(GS_indices):
        state_0 = basis.state(GS_bra_1)
        for i in range(N):
            basis.flip(state_0, i)
            state_1_index = basis.index(state_0)
            if state_1_index not in GS_indices:
                energy_gap_1 = GS_energy - state_energy(basis, Jij, state_1_index)
                for j in range(N):
                    basis.flip(state_0, j)
                    state_2_index = basis.index(state_0)
                    if state_2_index not in GS_indices:
                        energy_gap_2 = GS_energy - state_energy(basis, Jij, state_2_index)
                        for k in range(N):
                            basis.flip(state_0, k)
                            state_3_index = basis.index(state_0)
                            GS_3_index = np.argwhere(np.array(GS_indices) == state_3_index)
                            if len(GS_3_index) > 0:
                                row = GS_3_index[0][0]
                                H_app_3[row, column] += 1.0/(energy_gap_1*energy_gap_2)
                            basis.flip(state_0, k)
                    basis.flip(state_0, j)
            basis.flip(state_0, i)
    return H_app_3

def H_app_1st(h_x, H_0, V):
    # Calculate final 1st order
    return H_0 - h_x*V

def H_app_2nd(h_x, H_0, V, H_2):
    # Calculate final 2nd order approximated matrix
    c_2 = h_x**2
    return H_0 - h_x*V + H_2*c_2

def H_app_3rd(h_x, H_0, V, H_2, H_3):
    # Calculate final 3rd approximated matrix
    c_2 = h_x**2
    c_3 = h_x**3
    return H_0 - h_x*V + H_2*c_2 - H_3*c_3

def H_app_4th(h_x, H_0, V, H_2, H_3, H_4):
    # Calculate final 3rd approximated matrix
    c_2 = h_x**2
    c_3 = h_x**3
    c_4 = h_x**4
    return H_0 - h_x*V + H_2*c_2 - H_3*c_3 + H_4*c_4

def V_exact(basis, lattice):
    V_exact = np.zeros((basis.M, basis.M))
    for ket in range(basis.M):
        state = basis.state(ket)
        for i in range(lattice.N):
            basis.flip(state,i)
            bra = basis.index(state)
            V_exact[bra, ket] += 1
            basis.flip(state,i)
    return V_exact

def H_0_exact(Energies):
    return np.diag(Energies)

def H_exact(h_x, V_exact, H_0_exact):
    # Calculate exact matrix
    return H_0_exact - h_x*V_exact

###############################################################################

def app_1_eigensystem(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))

    H_0 = H_app_0(GS_energy, GS_indices)
    V = H_app_1(basis, GS_indices, N)

    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = np.linalg.eigh(H_app_1st(h_x, H_0, V, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, V

def app_2_eigensystem(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
    H_0 = H_app_0(GS_energy, GS_indices)
    V = H_app_1(basis, GS_indices, N)
    H_2 = H_app_2(basis, Jij, GS_indices, N, GS_energy)

    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = np.linalg.eigh(H_app_2nd(h_x, H_0, V, H_2, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_2

def app_2_eigensystem_general_matrices(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
    ES_1_indices = tfim_matrices.Hamming_set(basis, GS_indices, N, GS_indices)
    PVP = tfim_matrices.PVP(basis, GS_indices, N)
    PVQ = tfim_matrices.PVQ_1(basis, Jij, GS_indices, ES_1_indices, N, GS_energy)
    QVP = np.transpose(PVQ)
    energy_gap_matrix = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 1)
    
    # Build 0th order approximated matrix
    H_0 = H_app_0(GS_energy, GS_indices)

    # Start building the 1st order Hamiltonian
    H_app_1 = PVP

    # Start building the 2nd Hamiltonian
    H_app_2 = PVQ @ energy_gap_matrix @ QVP
    
    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = np.linalg.eigh(H_app_2nd(h_x, H_0, H_app_1, H_app_2, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_app_2

def app_3_eigensystem(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
    H_0 = H_app_0(GS_energy, GS_indices)
    V = H_app_1(basis, GS_indices, N)
    H_2 = H_app_2(basis, Jij, GS_indices, N, GS_energy)
    H_3 = H_app_3(basis, Jij, GS_indices, N, GS_energy)
    
    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = eigh(H_app_3rd(h_x, H_0, V, H_2, H_3, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_3

def app_3_eigensystem_general_matrices(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
        # Building blocks matrices
    ES_1_indices = tfim_matrices.Hamming_set(basis, GS_indices, N, GS_indices)
    PVP = tfim_matrices.PVP(basis, GS_indices, N)
    PVQ1 = tfim_matrices.PVQ_1(basis, Jij, GS_indices, ES_1_indices, N, GS_energy)
    Q1VP = np.transpose(PVQ1)
    Q1VQ1 = tfim_matrices.Q_1VQ_1(basis, ES_1_indices, GS_indices, N)

    # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
    EGM_12 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 2)
    EGM_13 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 3)
    EGM_11 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 1)

    # Start building Hamiltonians
    H_0 = H_app_0(GS_energy, GS_indices)

    H_app_1 = PVP

    H_app_2 = PVQ1 @ EGM_11 @ Q1VP

    H_app_3 = -0.5*(PVP @ PVQ1 @ EGM_12 @ Q1VP + np.transpose(PVP @ PVQ1 @ EGM_12 @ Q1VP)) + PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_11 @ Q1VP
    
    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = eigh(H_app_3rd(h_x, H_0, H_app_1, H_app_2, H_app_3, J));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_app_3

def app_4_eigensystem_general_matrices(GS_indices, GS_energy, h_x_range, J, N, basis, Jij):
    # Calculate approximated eigenvalues and eigenstates for range(h_x)
    app_eigenvalues = np.zeros((len(GS_indices), len(h_x_range)))
    app_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    
    # Building blocks matrices
    ES_1_indices = tfim_matrices.Hamming_set(basis, GS_indices, N, GS_indices)
    ES_2_indices = tfim_matrices.Hamming_set(basis, ES_1_indices, N, GS_indices)
    
    # Building blocks matrices
    PVP = tfim_matrices.PVP(basis, GS_indices, N)
    PVQ1 = tfim_matrices.PVQ_1(basis, Jij, GS_indices, ES_1_indices, N, GS_energy)
    Q1VP = np.transpose(PVQ1)
    Q1VQ1 = tfim_matrices.Q_1VQ_1(basis, ES_1_indices, GS_indices, N)
    Q1VQ2 = tfim_matrices.Q_1VQ_2(basis, ES_2_indices, ES_1_indices, GS_indices, N)
    Q2VQ1 = np.transpose(Q1VQ2)

    # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
    EGM_12 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 2)
    EGM_13 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 3)
    EGM_11 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, N, GS_energy, 1)
    EGM_21 = tfim_matrices.energy_gap(basis, Jij, ES_2_indices, N, GS_energy, 1)
    
    # Start building Hamiltonians

    H_0 = H_app_0(GS_energy, GS_indices)

    H_app_1 = PVP

    H_app_2 = PVQ1 @ EGM_11 @ Q1VP

    H_app_3 = -0.5*(PVP @ PVQ1 @ EGM_12 @ Q1VP + np.transpose(PVP @ PVQ1 @ EGM_12 @ Q1VP)) + PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_11 @ Q1VP
    
    H_app_4 = 0.5*(tfim_matrices.hc(PVQ1 @ EGM_13 @ Q1VP @ PVP @ PVP)) - 0.5*(tfim_matrices.hc(PVQ1 @ EGM_12 @ Q1VP @ PVQ1 @ EGM_11 @ Q1VP)) - 1.*(tfim_matrices.hc(PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_12 @ Q1VP @ PVP)) + 1.*(PVQ1 @ EGM_11 @ Q1VQ2 @ EGM_21 @ Q2VQ1 @ EGM_11 @ Q1VP)

    for j, h_x in enumerate(h_x_range):
        app_eigenvalue, app_eigenstate = eigh(H_app_4th(h_x, H_0, H_app_1, H_app_2, H_app_3, H_app_4));
        for i in range(len(GS_indices)):
            app_eigenvalues[i][j] = app_eigenvalue[i]
            for k in range(len(GS_indices)):
                app_eigenstates[j][i][k] = app_eigenstate[i][k]
    return app_eigenvalues, app_eigenstates, H_app_4

def exc_eigensystem(basis, h_x_range, lattice, Energies):
    # Calculate exact eigenvalues and eigenstates for range(h_x)
    exc_eigenvalues = np.zeros((basis.M, len(h_x_range)))
    exc_eigenstates = np.zeros((len(h_x_range), basis.M, basis.M))
    V_exc = V_exact(basis, lattice)
    H_0_exc = H_0_exact(Energies)
    for j, h_x in enumerate(h_x_range):
        exc_eigenvalue, exc_eigenstate = np.linalg.eigh(H_exact(h_x, V_exc, H_0_exc));
        for i in range(basis.M):
            exc_eigenvalues[i][j] = exc_eigenvalue[i]
            for k in range(basis.M):
                exc_eigenstates[j][i][k] = exc_eigenstate[i][k]
    return exc_eigenvalues, exc_eigenstates

# modified exact Hamiltonians using compressed sparse row matrices
def V_exact_csr(basis, lattice):
    row = []
    col = []
    N = lattice.N
    for ket in range(basis.M):
        state = basis.state(ket)
        for i in range(lattice.N):
            basis.flip(state, i)
            bra = basis.index(state)
            row.append(bra)
            col.append(ket)
            basis.flip(state, i)
    data = np.ones(len(col))
    V_exact = sparse.csr_matrix((data, (np.array(row), np.array(col))), shape=(2 ** N, 2 ** N))
    return V_exact

def H_0_exact_csr(Energies):
    return sparse.diags(Energies)

###############################################################################
# For error analysis and curve fit

def poly_5(x, b):
    return b*x**5;

def poly_4(x, b):
    return b*x**4.;

def poly_3(x, b):
    # third order polynomial
    return b*x**3;

def poly_2(x, a):
    # second order polynomial
    return a*x**2;

def prob(eigenstate):
    norm = np.vdot(eigenstate, eigenstate)
    normed_eigenstate = eigenstate/(norm**0.5)
    return np.conjugate(normed_eigenstate)*normed_eigenstate

def prob_app(GS_indices, h_x_range, app_eigenstates):
    # Calculate probabilities for approximated eigenstates
    prob_app = np.zeros((len(GS_indices), len(h_x_range),))
    for j, h_x in enumerate(h_x_range):
        GS_prob_vector = prob(app_eigenstates[j][:, 0])
        for i in range(len(GS_indices)):
            prob_app[i][j] = GS_prob_vector[i]
    return prob_app

def prob_exc(GS_indices, h_x_range, exc_eigenstates, index):
    # Calculate probabilities for exact eigenstates
    prob_exc = np.zeros((len(GS_indices), len(h_x_range)))
    for j, h_x in enumerate(h_x_range):
        GS_prob_vector = prob(exc_eigenstates[j][:, index])
        for i in range(len(GS_indices)):
            prob_exc[i][j] = GS_prob_vector[i]
    return prob_exc

def prob_exc_total(GS_indices, h_x_range, exc_eigenstates):
    # Probability of finding the system to be in each of the ground states
    prob_exc_total = np.zeros((len(GS_indices), len(GS_indices), len(h_x_range),))
    for n in range(len(GS_indices)):
        for j, h_x in enumerate(h_x_range):
            GS_prob_vector = prob(np.transpose(exc_eigenstates[j])[n])
            for k, i in enumerate(GS_indices):
                prob_exc_total[n][k][j] = GS_prob_vector[i]
    return prob_exc_total

def prob_excited_sum(GS_indices, h_x_range, prob_exc_total):
    # Probability of finding the system to be in excited states
    prob_excited_sum = np.zeros((len(GS_indices), len(h_x_range)))
    for n in range(len(GS_indices)):
        for i, h_x in enumerate(h_x_range):
            prob_excited_sum[n][i] = 1 - np.sum(prob_exc_total[n][:, i])
    return prob_excited_sum

def normalize(eigenstate):
    # Normalize
    norm = np.vdot(eigenstate, eigenstate)**0.5
    return eigenstate/norm

def GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates):
    GS_exc_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    for j in range(len(h_x_range)):
        for n, m in enumerate(GS_indices):
            for i, k in enumerate(GS_indices):
                GS_exc_eigenstates[j, n, i] = exc_eigenstates[j, m, i]
    return GS_exc_eigenstates

def norm_GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates):
    # Renormalize
    normalized_GS_exc_eigenstates = np.zeros((len(h_x_range), len(GS_indices), len(GS_indices)))
    GS_exc_ES = GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates)
    for j in range(len(h_x_range)):
        for n in range(len(GS_indices)):
            normed_vector = normalize(GS_exc_ES[j, :, n])
            for i in range(len(GS_indices)):
                normalized_GS_exc_eigenstates[j, i, n] = normed_vector[i]
    return normalized_GS_exc_eigenstates

def fidelity(exc_eigenstate, app_eigenstate):
    # Calculate fidelity
    dot = np.vdot(exc_eigenstate, np.conjugate(app_eigenstate))
    return dot*np.conjugate(dot)

def sort(lst):
    # identify degenerate energy level and resort
    epsilon = 1*10**(-12)
    order = [];
    floor = np.array([0]);
    for i in range(1, len(lst)):
        if abs(lst[i-1] - lst[i]) <= epsilon:
            floor = np.append(floor, i)
        else:  
            order.append(floor)
            floor = np.array([i]);
    order.append(floor)
    return list(order) 

def fidelity_array(GS_indices, h_x_range, GS_exc_eigenvalues, app_eigenvalues, exc_eigenstates, app_eigenstates):
    # Produce an array of fidelities between exc and app to be plotted
    fidelity_array = np.zeros((len(GS_indices), len(h_x_range)))
    GS_exc_ES = GS_exc_eigenstates(GS_indices, h_x_range, exc_eigenstates)
    for i in range(len(h_x_range)):
        sorted_exc_energy_indices = sort(GS_exc_eigenvalues[:, i])
        sorted_app_energy_indices = sort(app_eigenvalues[:, i])
        for l, app_level in enumerate(sorted_app_energy_indices):
            for j in app_level:
                fidel_sum = 0;
                for k in sorted_exc_energy_indices[l]:
                    fidel = fidelity(GS_exc_ES[i, :, k], app_eigenstates[i, :, j])
                    fidel_sum += fidel
                fidelity_array[j, i] = fidel_sum
    return fidelity_array

def infidelity_array(fidelity_array):
    return 1 - fidelity_array


# building 2D NN Jij matrices
def Jij_2D_NN(seed, N, PBC, xwidth, yheight, lattice):
    def bond_list(seed, N, PBC, xwidth, yheight):
        np.random.seed(seed)
        # Generates a random list of bonds with equal numbers of ferromagnetic and antiferromagnetic bonds
        if PBC == True:
            num_of_bonds = 2 * N
        else:
            num_of_bonds = (xwidth - 1) * (yheight) + (xwidth) * (yheight - 1)
        if num_of_bonds % 2 == 0:
            a1 = [-1 for i in range(num_of_bonds // 2)]
        else:
            a1 = [-1 for i in range((num_of_bonds // 2) + 1)]
        a2 = [1 for i in range(num_of_bonds // 2)]
        a = list(np.random.permutation(a1 + a2))
        return a

    def make_Jij(N, b_list, lattice):
        # Goes through the list of bonds to make the jij matrix that tells you how all of the spins are bonded to each other

        bond_index = 0
        Jij = np.zeros((N, N))
        for i in range(0, N):
            NNs = lattice.NN(i)
            for j in NNs:
                if Jij[i][j] == 0:
                    Jij[i][j] = b_list[bond_index]
                    Jij[j][i] = b_list[bond_index]
                    bond_index += 1
        return Jij

    b_list = bond_list(seed, N, PBC, xwidth, yheight)

    def Jij_convert(Jij, N):
        new = np.zeros((N // 2, N))
        for j in range(0, N):  # columns
            count = 0
            while count < N // 2:
                if j < N // 2:
                    # make i at the index J be N-1
                    subtract = 1
                    for i in range(j, N // 2):
                        new[i][j] = Jij[j][N - subtract]
                        subtract += 1
                        count += 1
                        if count == N // 2:
                            break
                i = j
                start = 0
                while i > N // 2:
                    i -= 1
                    start += 1
                for q in range(0, N // 2):
                    new[i - 1][j] = Jij[j][q + start]
                    count += 1
                    i -= 1
                    if i <= 0:
                        break
        return new

    Jij = make_Jij(N, b_list, lattice)

    return Jij_convert(Jij, N), Jij


def eight_tile(seed, p):
    np.random.seed(seed)
    bond_pos_lst = np.array([[1,6], [1,8], [1,2], [1,4], [2,7],[2,3], [2,5], [3,8], [3,6], [3,4], [4,5], [4,7], [6,7], [8,7], [8,5], [5,6]])
    i = [np.random.random() for _ in range(np.shape(bond_pos_lst)[0])]
    a = np.zeros(len(i))
    for index, prob_seed in enumerate(i):
        if prob_seed <= p:
            a[index] += 1
        else:
            a[index] -= 1
    Jij = np.zeros((8,8))
    for i, pos in enumerate(bond_pos_lst):
        Jij[pos[0]-1,pos[1]-1] += a[i]
        Jij[pos[1]-1,pos[0]-1] += a[i]
        
    N = 8
    
    return Jij, N

def ten_tile(seed, p):
    np.random.seed(seed)
    bond_pos_lst = np.array([[1,10], [1,8], [1,2], [1,4], [2,9], [2,3], [2,5], [3,10], [3,4], [3,6], [4,5], [4,7], [5,6], [5,8], [6,7], [6,9], [7,8], [7,10], [8,9], [9,10]])
    i = [np.random.random() for _ in range(np.shape(bond_pos_lst)[0])]
    a = np.zeros(len(i))
    for index, prob_seed in enumerate(i):
        if prob_seed <= p:
            a[index] += 1
        else:
            a[index] -= 1
    Jij = np.zeros((10, 10))
    for i, pos in enumerate(bond_pos_lst):
        Jij[pos[0]-1,pos[1]-1] += a[i]
        Jij[pos[1]-1,pos[0]-1] += a[i]
    N = 10
    return Jij, N

###############################################################################
# the following functions enables susceptibility calculation using perturbation theory

def sigma_z_0(a, input_state_indices, basis):
    '''
    Builds Pauli spin z matrix for the a-th spin, projected onto the ground subspace
    '''
    sigma_z_a = np.zeros(len(input_state_indices))
    for state_index, ket in enumerate(input_state_indices):
        state = basis.state(ket)
        if state[a] == 1:
            sigma_z_a[state_index] += 1
        else:
            sigma_z_a[state_index] -= 1
    return sigma_z_a


def susceptibility(h_x_range, lattice, basis, exc_eigenvalues, H_0_exc, V_exc, v0, h_z):
    order_param_matrix = np.zeros((len(h_x_range), lattice.N))
    chi_aa_matrix = np.zeros((len(h_x_range), lattice.N))
    E1_arr = np.zeros(len(h_x_range))
    for i, h_x in enumerate(h_x_range):
        for a in range(lattice.N):
            sigma_z = np.zeros(basis.M)
            for ket in range(basis.M):
                state = basis.state(ket)
                if state[a] == 1:
                    sigma_z[ket] += 1
                else:
                    sigma_z[ket] -= 1
            E0 = exc_eigenvalues[i]
            E1 = spla.eigsh(H_0_exc - V_exc.multiply(h_x) - h_z * sparse.diags(sigma_z), k=1, which='SA', v0=v0, maxiter=200,
                       return_eigenvectors=False)[0]
            E2 = spla.eigsh(H_0_exc - V_exc.multiply(h_x) - 2. * h_z * sparse.diags(sigma_z), k=1, which='SA', v0=v0,
                            maxiter=200, return_eigenvectors=False)[0]
            E1_arr[i] = E1
            order_param_matrix[i, a] = (E2 - 4. * E1 + 3. * E0) / (-2. * h_z)
            chi_aa = 2 * (E2 - 2. * E1 + E0) / (2 * h_z ** 2.)
            chi_aa_matrix[i, a] = chi_aa

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
            for b in range(a + 1, lattice.N, 1):
                sigma_z_b = np.zeros(basis.M)
                for ket in range(basis.M):
                    state = basis.state(ket)
                    if state[b] == 1:
                        sigma_z_b[ket] = 1
                    else:
                        sigma_z_b[ket] = 1
                H1 = H_0_exc - V_exc.multiply(h_x) - (sparse.diags(sigma_z_a) + sparse.diags(sigma_z_b)).multiply(h_z)
                H2 = H_0_exc - V_exc.multiply(h_x) - (sparse.diags(sigma_z_a) + sparse.diags(sigma_z_b)).multiply(
                    2. * h_z)
                E0 = exc_eigenvalues[i]
                E1 = spla.eigsh(H1, k=1, which='SA', v0=v0,
                                maxiter=200, return_eigenvectors=False)[0]
                E2 = spla.eigsh(H2, k=1, which='SA', v0=v0,
                           maxiter=200, return_eigenvectors=False)[0]
                chi_ab = (E2 - 2. * E1 + E0) / (2 * (h_z ** 2.)) - 0.5 * (chi_aa_matrix[i, a] + chi_aa_matrix[i, b])
                chi_ab_matrix[i, a, b] = chi_ab
                chi_ab_matrix[i, b, a] = chi_ab
                # adding the diagonal elements
                for c in range(lattice.N):
                    chi_ab_matrix[i, c, c] = chi_aa_matrix[i, c]

    chi_arr = np.zeros(len(h_x_range))
    for i, h_x in enumerate(h_x_range):
        chi_arr[i] += np.sum(chi_ab_matrix[i])

    order_param_arr = np.zeros(len(h_x_range))
    for i, h_x in enumerate(h_x_range):
        order_param_arr[i] += np.sum(abs(order_param_matrix[i]))

    return chi_arr, order_param_arr

def energy_gap_longitudinal_1(a, basis, Jij, lower_excitation_states, higher_excitation_states, exponent, h_z):
    energy_gap = np.zeros((len(lower_excitation_states), len(higher_excitation_states)))
    sigma_high = sigma_z_0(a, higher_excitation_states, basis)
    sigma_low = sigma_z_0(a, lower_excitation_states, basis)
    for i, lower_state_index in enumerate(lower_excitation_states):
        for j, higher_state_index in enumerate(higher_excitation_states):
            longitudinal_field_lower = - h_z * sigma_low[i]
            longitudinal_field_higher = - h_z * sigma_high[j]
            ground_energy = state_energy(basis, Jij, lower_state_index) + longitudinal_field_lower
            excited_energy = state_energy(basis, Jij, higher_state_index) + longitudinal_field_higher
            energy_gap[i, j] = 1./(ground_energy - excited_energy) ** exponent
    return energy_gap

def energy_gap_longitudinal_2(a, b, basis, Jij, lower_excitation_states, higher_excitation_states, exponent, h_z):
    energy_gap = np.zeros((len(lower_excitation_states), len(higher_excitation_states)))
    sigma_high_a = sigma_z_0(a, higher_excitation_states, basis)
    sigma_low_a = sigma_z_0(a, lower_excitation_states, basis)
    sigma_high_b = sigma_z_0(b, higher_excitation_states, basis)
    sigma_low_b = sigma_z_0(b, lower_excitation_states, basis)
    for i, lower_state_index in enumerate(lower_excitation_states):
        for j, higher_state_index in enumerate(higher_excitation_states):
            longitudinal_field_lower = - h_z * (sigma_low_a[i] + sigma_low_b[i])
            longitudinal_field_higher = - h_z * (sigma_high_a[j] + sigma_high_b[j])
            ground_energy = state_energy(basis, Jij, lower_state_index) + longitudinal_field_lower
            excited_energy = state_energy(basis, Jij, higher_state_index) + longitudinal_field_higher
            energy_gap[i, j] = 1./(ground_energy - excited_energy) ** exponent
    return energy_gap

def fourth_order_last_term_single(a, basis, Jij, GS_indices, ES_1_indices, ES_2_indices, N, h_z):
    H = np.zeros((len(GS_indices), len(GS_indices)))
    for column, GS_bra_1 in enumerate(GS_indices):
        state_0 = basis.state(GS_bra_1)
        sigma_0 = sigma_z_0(a, GS_indices, basis)
        LE = state_energy(basis, Jij, GS_bra_1) - h_z * sigma_0[column]
        for i in range(N):
            basis.flip(state_0, i)
            state_1_index = basis.index(state_0)
            if state_1_index in ES_1_indices:
                sigma_1 = sigma_z_0(a, ES_1_indices, basis)
                HE = state_energy(basis, Jij, state_1_index) - h_z * sigma_1[np.argwhere(ES_1_indices == state_1_index)]
                EG_1 = LE - HE
                for j in range(N):
                    basis.flip(state_0, j)
                    state_2_index = basis.index(state_0)
                    if state_2_index in ES_2_indices:
                        sigma_2 = sigma_z_0(a, ES_2_indices, basis)
                        HE = state_energy(basis, Jij, state_2_index) - h_z * sigma_2[np.argwhere(ES_2_indices == state_2_index)]
                        EG_2 = LE - HE
                        for k in range(N):
                            basis.flip(state_0, k)
                            state_3_index = basis.index(state_0)
                            if state_3_index in ES_1_indices:
                                HE = state_energy(basis, Jij, state_3_index) - h_z * sigma_1[np.argwhere(ES_1_indices == state_3_index)]
                                EG_3 = LE - HE
                                for l in range(N):
                                    basis.flip(state_0, l)
                                    state_4_index = basis.index(state_0)
                                    if state_4_index in GS_indices:
                                        row = np.argwhere(np.array(GS_indices) == state_4_index)[0][0]
                                        H[row, column] += 1./(EG_1 * EG_2 * EG_3)
                                    basis.flip(state_0, l)
                            basis.flip(state_0, k)
                    basis.flip(state_0, j)
            basis.flip(state_0, i)
    return H

def fourth_order_last_term_double(a, b, basis, Jij, GS_indices, ES_1_indices, ES_2_indices, N, h_z):
    H = np.zeros((len(GS_indices), len(GS_indices)))
    for column, GS_bra_1 in enumerate(GS_indices):
        state_0 = basis.state(GS_bra_1)
        sigma_0_a = sigma_z_0(a, GS_indices, basis)
        sigma_0_b = sigma_z_0(b, GS_indices, basis)
        LE = state_energy(basis, Jij, GS_bra_1) - h_z * (sigma_0_a + sigma_0_b)[column]
        for i in range(N):
            basis.flip(state_0, i)
            state_1_index = basis.index(state_0)
            if state_1_index in ES_1_indices:
                sigma_1_a = sigma_z_0(a, ES_1_indices, basis)
                sigma_1_b = sigma_z_0(b, ES_1_indices, basis)
                HE = state_energy(basis, Jij, state_1_index) - h_z * (sigma_1_a + sigma_1_b)[np.argwhere(ES_1_indices == state_1_index)]
                EG_1 = LE - HE
                for j in range(N):
                    basis.flip(state_0, j)
                    state_2_index = basis.index(state_0)
                    if state_2_index in ES_2_indices:
                        sigma_2_a = sigma_z_0(a, ES_2_indices, basis)
                        sigma_2_b = sigma_z_0(b, ES_2_indices, basis)
                        HE = state_energy(basis, Jij, state_2_index) - h_z * (sigma_2_a + sigma_2_b)[np.argwhere(ES_2_indices == state_2_index)]
                        EG_2 = LE - HE
                        for k in range(N):
                            basis.flip(state_0, k)
                            state_3_index = basis.index(state_0)
                            if state_3_index in ES_1_indices:
                                HE = state_energy(basis, Jij, state_3_index) - h_z * (sigma_1_a + sigma_1_b)[np.argwhere(ES_1_indices == state_3_index)]
                                EG_3 = LE - HE
                                for l in range(N):
                                    basis.flip(state_0, l)
                                    state_4_index = basis.index(state_0)
                                    if state_4_index in GS_indices:
                                        row = np.argwhere(np.array(GS_indices) == state_4_index)[0][0]
                                        H[row, column] += 1./(EG_1 * EG_2 * EG_3)
                                    basis.flip(state_0, l)
                            basis.flip(state_0, k)
                    basis.flip(state_0, j)
            basis.flip(state_0, i)
    return H

def perturb_susceptibility(h_x_range, L, PBC, Jij, GS_indices, h_z):
    '''
    calculates spin glass susceptibility at low perturbation using perturbation theory, projected fast algorithm but lacking benchmark; like entanglement entropy,
    this algorithm should only be used for low perturbations.
    '''
    # initiating output array
    chi_arr_all = []
    order_param_all = []

    lattice = tfim.Lattice(L, PBC)
    N = lattice.N
    basis = tfim.IsingBasis(lattice)
    GS_energy = state_energy(basis, Jij, GS_indices[0])

    #########################################################################

    # construct building blocks
    ES_1_indices = tfim_matrices.Hamming_set(basis, GS_indices, N, GS_indices)
    ES_2_indices = tfim_matrices.Hamming_set(basis, ES_1_indices, N, GS_indices)

    # Building blocks matrices
    PVP = tfim_matrices.PVP(basis, GS_indices, N)
    PVQ1 = tfim_matrices.PVQ_1(basis, GS_indices, ES_1_indices, N)
    Q1VP = np.transpose(PVQ1)
    Q1VQ1 = tfim_matrices.Q_1VQ_1(basis, ES_1_indices, GS_indices, N)
    Q1VQ2 = tfim_matrices.Q_1VQ_2(basis, ES_2_indices, ES_1_indices, GS_indices, N)
    Q2VQ1 = np.transpose(Q1VQ2)

    # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
    EGM_12 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, GS_energy, 2)
    EGM_13 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, GS_energy, 3)
    EGM_11 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, GS_energy, 1)
    EGM_21 = tfim_matrices.energy_gap(basis, Jij, ES_2_indices, GS_energy, 1)

    # Start building Hamiltonians

    H_0 = H_app_0(GS_energy, GS_indices)
    H_app_1 = PVP
    H_app_2 = PVQ1 @ EGM_11 @ Q1VP
    H_app_3 = -0.5 * (PVP @ PVQ1 @ EGM_12 @ Q1VP + np.transpose(
        PVP @ PVQ1 @ EGM_12 @ Q1VP)) + PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_11 @ Q1VP
    H_app_4 = 0.5 * (tfim_matrices.hc(PVQ1 @ EGM_13 @ Q1VP @ PVP @ PVP)) - 0.5 * (
        tfim_matrices.hc(PVQ1 @ EGM_12 @ Q1VP @ PVQ1 @ EGM_11 @ Q1VP)) - 1. * (
                  tfim_matrices.hc(PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_12 @ Q1VP @ PVP)) + 1. * (
                          PVQ1 @ EGM_11 @ Q1VQ2 @ EGM_21 @ Q2VQ1 @ EGM_11 @ Q1VP)

##############################################################################
    # construct connectivity matrix
    conn_matrix = H_0 + H_app_1 + H_app_2 + H_app_3 + H_app_4
    adj_matrix = np.zeros(np.shape(conn_matrix))
    for m in range(len(GS_indices)):
        for n in range(len(GS_indices)):
            if conn_matrix[m, n] != 0.:
                adj_matrix[m, n] = 1
    G = nx.from_numpy_matrix(adj_matrix)
    connectivity = nx.is_connected(G)
    print("number of ground states: ", len(GS_indices), 'connected:', connectivity)
##############################################################################

    chi_arr = np.zeros(len(h_x_range))
    order_param_arr = np.zeros(len(h_x_range))
    if connectivity:
        for l, h_x in enumerate(h_x_range):
            # E0 is the original ground energy without applying the longitudinal field
            E0 = eigh(H_app_4th(h_x, H_0, H_app_1, H_app_2, H_app_3, H_app_4), eigvals_only = True)[0]

            # calculate chi_aa
            chi_aa_matrix = np.zeros(lattice.N)
            order_param_matrix = np.zeros(lattice.N)
            for a in range(N):
                '''
                recalculate effective Hamiltonian as the longitudinal field has caused Ising energy splitting
                '''

                # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
                EGM_12 = energy_gap_longitudinal_1(a, basis, Jij, GS_indices, ES_1_indices, 2, h_z)
                EGM_13 = energy_gap_longitudinal_1(a, basis, Jij, GS_indices, ES_1_indices, 3, h_z)
                EGM_11 = energy_gap_longitudinal_1(a, basis, Jij, GS_indices, ES_1_indices, 1, h_z)
                EGM_21 = energy_gap_longitudinal_1(a, basis, Jij, GS_indices, ES_2_indices, 1, h_z)

                # Start building Hamiltonians

                H_0 = H_app_0(GS_energy, GS_indices)
                H_app_1 = PVP
                H_app_2 = (PVQ1 * EGM_11) @ Q1VP
                H_app_3 = -0.5 * (PVP @ (PVQ1 * EGM_12) @ Q1VP + np.transpose(
                    PVP @ (PVQ1 * EGM_12) @ Q1VP)) + (PVQ1 * EGM_11) @ Q1VQ1 @ (EGM_11.T * Q1VP)
                H_app_4 = 0.5 * (tfim_matrices.hc((PVQ1 * EGM_13) @ Q1VP @ PVP @ PVP)) - 0.5 * (
                    tfim_matrices.hc((PVQ1 * EGM_12) @ Q1VP @ (PVQ1 * EGM_11) @ Q1VP)) - 1. * (tfim_matrices.hc((PVQ1 * EGM_11) @ Q1VQ1 @ (EGM_12.T * Q1VP) @ PVP)) + fourth_order_last_term_single(a, basis, Jij, GS_indices, ES_1_indices, ES_2_indices, N)

                # build Pauli matrices
                sigma_z_a = sigma_z_0(a, GS_indices, basis)

                # calculating susceptibility
                E1 = eigh(H_app_4th(h_x, H_0 - h_z * sigma_z_a, H_app_1, H_app_2, H_app_3, H_app_4), eigvals_only = True)[0]
                E2 = eigh(H_app_4th(h_x, H_0 - 2. * h_z * sigma_z_a, H_app_1, H_app_2, H_app_3, H_app_4), eigvals_only = True)[0]
                order_param = (E2 - 4. * E1 + 3. * E0) / (-2. * h_z)
                order_param_matrix[a] = order_param
                chi_aa = 2 * (E2 - 2. * E1 + E0) / (2 * h_z ** 2.)
                chi_aa_matrix[a] = chi_aa
                print('chi_aa_matrix ', a, 'completed')
            # calculate chi_ab
            chi_ab_matrix = np.zeros((lattice.N, lattice.N))
            for a in range(lattice.N):
                for b in range(a + 1, lattice.N, 1):
                    '''
                    recalculate effective Hamiltonian as the longitudinal field has caused Ising energy splitting
                    '''
                    # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
                    EGM_12 = energy_gap_longitudinal_2(a, b, basis, Jij, GS_indices, ES_1_indices, 2, h_z)
                    EGM_13 = energy_gap_longitudinal_2(a, b, basis, Jij, GS_indices, ES_1_indices, 3, h_z)
                    EGM_11 = energy_gap_longitudinal_2(a, b, basis, Jij, GS_indices, ES_1_indices, 1, h_z)
                    EGM_21 = energy_gap_longitudinal_2(a, b, basis, Jij, GS_indices, ES_2_indices, 1, h_z)

                    # Start building Hamiltonians

                    H_0 = H_app_0(GS_energy, GS_indices)
                    H_app_1 = PVP
                    H_app_2 = (PVQ1 * EGM_11) @ Q1VP
                    H_app_3 = -0.5 * (PVP @ (PVQ1 * EGM_12) @ Q1VP + np.transpose(
                    PVP @ (PVQ1 * EGM_12) @ Q1VP)) + (PVQ1 * EGM_11) @ Q1VQ1 @ (EGM_11.T * Q1VP)
                    H_app_4 = 0.5 * (tfim_matrices.hc((PVQ1 * EGM_13) @ Q1VP @ PVP @ PVP)) - 0.5 * (
                        tfim_matrices.hc((PVQ1 * EGM_12) @ Q1VP @ (PVQ1 * EGM_11) @ Q1VP)) - 1. * (tfim_matrices.hc((PVQ1 * EGM_11) @ Q1VQ1 @ (EGM_12.T * Q1VP) @ PVP)) + fourth_order_last_term_double(a, b, basis, Jij, GS_indices, ES_1_indices, ES_2_indices, N, h_z)

                    # build Pauli matrices
                    sigma_z_a = sigma_z_0(a, GS_indices, basis)
                    sigma_z_b = sigma_z_0(b, GS_indices, basis)

                    # calculating suscepbility
                    H1 = H_app_4th(h_x, H_0 - h_z * (np.diag(sigma_z_a + sigma_z_b)), H_app_1, H_app_2, H_app_3, H_app_4)
                    H2 = H_app_4th(h_x, H_0 - 2. * h_z * (np.diag(sigma_z_a + sigma_z_b)), H_app_1, H_app_2, H_app_3, H_app_4)
                    E1 = eigh(H1, eigvals_only = True)[0]
                    E2 = eigh(H2, eigvals_only = True)[0]
                    chi_ab = (E2 - 2. * E1 + E0) / (2 * (h_z ** 2.)) - 0.5 * (chi_aa_matrix[a] + chi_aa_matrix[b])
                    chi_ab_matrix[a, b] = chi_ab
                    chi_ab_matrix[b, a] = chi_ab

                    print('chi_ab ', a, b, 'completed')

            # adding the diagonal elements
            for c in range(lattice.N):
                chi_ab_matrix[c, c] = chi_aa_matrix[c]

            print('h_x = ', h_x, 'completed')

            chi, order_param = np.sum(np.power(chi_ab_matrix, 2.)), abs(np.sum(order_param_matrix))
            chi_arr[l] = chi
            order_param_arr[l] = order_param

    else:
        C = max(nx.connected_components(G))
        S = G.subgraph(C)
        for l, h_x in enumerate(h_x_range):
            # E0 is the original ground energy without applying the longitudinal field
            E0 = eigh(H_app_4th(h_x, H_0, H_app_1, H_app_2, H_app_3, H_app_4), eigvals_only = True)[0]
            # calculate chi_aa
            chi_aa_matrix = np.zeros(lattice.N)
            order_param_matrix = np.zeros(lattice.N)
            for a in range(N):
                '''
                recalculate effective Hamiltonian as the longitudinal field has caused Ising energy splitting
                '''
                # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
                EGM_12 = energy_gap_longitudinal_1(a, basis, Jij, GS_indices, ES_1_indices, 2, h_z)
                EGM_13 = energy_gap_longitudinal_1(a, basis, Jij, GS_indices, ES_1_indices, 3, h_z)
                EGM_11 = energy_gap_longitudinal_1(a, basis, Jij, GS_indices, ES_1_indices, 1, h_z)
                EGM_21 = energy_gap_longitudinal_1(a, basis, Jij, GS_indices, ES_2_indices, 1, h_z)

                # Start building Hamiltonians

                H_0 = H_app_0(GS_energy, GS_indices)
                H_app_1 = PVP
                H_app_2 = (PVQ1 * EGM_11) @ Q1VP
                H_app_3 = -0.5 * (PVP @ (PVQ1 * EGM_12) @ Q1VP + np.transpose(
                    PVP @ (PVQ1 * EGM_12) @ Q1VP)) + (PVQ1 * EGM_11) @ Q1VQ1 @ (EGM_11.T * Q1VP)
                H_app_4 = 0.5 * (tfim_matrices.hc((PVQ1 * EGM_13) @ Q1VP @ PVP @ PVP)) - 0.5 * (
                        tfim_matrices.hc((PVQ1 * EGM_12) @ Q1VP @ (PVQ1 * EGM_11) @ Q1VP)) - 1. * (tfim_matrices.hc((PVQ1 * EGM_11) @ Q1VQ1 @ (EGM_12.T * Q1VP) @ PVP)) + fourth_order_last_term_single(a, basis, Jij, GS_indices, ES_1_indices, ES_2_indices, N, h_z)

                # build Pauli matrices
                sigma_z_a = sigma_z_0(a, GS_indices, basis)

                # effective Hamiltonian under longitudinal field

                H1 = H_app_4th(h_x, H_0 - h_z * sigma_z_a, H_app_1, H_app_2, H_app_3, H_app_4)
                H2 = H_app_4th(h_x, H_0 - 2. * h_z * sigma_z_a, H_app_1, H_app_2, H_app_3, H_app_4)
                # create new weighted adj_matrix (decoupled GS manifold)
                nodes = S.nodes()
                num_nodes = len(nodes)
                decoupled1 = np.zeros((num_nodes, num_nodes))
                decoupled2 = np.zeros((num_nodes, num_nodes))
                for p, node_index_1 in enumerate(nodes):
                    for q, node_index_2 in enumerate(nodes):
                        decoupled1[p, q] = H1[node_index_1, node_index_2]
                        decoupled2[p, q] = H2[node_index_1, node_index_2]

                # calculating susceptibility
                E1 = eigh(decoupled1, eigvals_only = True)[0]
                E2 = eigh(decoupled2, eigvals_only = True)[0]
                order_param = (E2 - 4. * E1 + 3. * E0) / (-2. * h_z)
                order_param_matrix[a] = order_param
                chi_aa = 2 * (E2 - 2. * E1 + E0) / (2 * h_z ** 2.)
                chi_aa_matrix[a] = chi_aa

                print('chi_aa ', a, 'completed')


            # calculate chi_ab
            chi_ab_matrix = np.zeros((lattice.N, lattice.N))
            for a in range(lattice.N):
                for b in range(a + 1, lattice.N, 1):
                    '''
                    recalculate effective Hamiltonian as the longitudinal field has caused Ising energy splitting
                    '''
                    # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
                    EGM_12 = energy_gap_longitudinal_2(a, b, basis, Jij, GS_indices, ES_1_indices, 2, h_z)
                    EGM_13 = energy_gap_longitudinal_2(a, b, basis, Jij, GS_indices, ES_1_indices, 3, h_z)
                    EGM_11 = energy_gap_longitudinal_2(a, b, basis, Jij, GS_indices, ES_1_indices, 1, h_z)
                    EGM_21 = energy_gap_longitudinal_2(a, b, basis, Jij, GS_indices, ES_2_indices, 1, h_z)

                    # Start building Hamiltonians

                    H_0 = H_app_0(GS_energy, GS_indices)

                    H_app_1 = PVP

                    H_app_2 = (PVQ1 * EGM_11) @ Q1VP

                    H_app_3 = -0.5 * (PVP @ (PVQ1 * EGM_12) @ Q1VP + np.transpose(
                    PVP @ (PVQ1 * EGM_12) @ Q1VP)) + (PVQ1 * EGM_11) @ Q1VQ1 @ (EGM_11.T * Q1VP)

                    H_app_4 = 0.5 * (tfim_matrices.hc((PVQ1 * EGM_13) @ Q1VP @ PVP @ PVP)) - 0.5 * (
                        tfim_matrices.hc((PVQ1 * EGM_12) @ Q1VP @ (PVQ1 * EGM_11) @ Q1VP)) - 1. * (tfim_matrices.hc((PVQ1 * EGM_11) @ Q1VQ1 @ (EGM_12.T * Q1VP) @ PVP)) + fourth_order_last_term_double(a, b, basis, Jij, GS_indices, ES_1_indices, ES_2_indices, N, h_z)
                    # build Pauli matrices
                    sigma_z_a = sigma_z_0(a, GS_indices, basis)
                    sigma_z_b = sigma_z_0(b, GS_indices, basis)

                    # calculating suscepbility
                    H1 = H_app_4th(h_x, H_0 - h_z * (np.diag(sigma_z_a + sigma_z_b)), H_app_1, H_app_2, H_app_3, H_app_4)
                    H2 = H_app_4th(h_x, H_0 - 2. * h_z * (np.diag(sigma_z_a + sigma_z_b)), H_app_1, H_app_2, H_app_3, H_app_4)

                    # create new weighted adj_matrix (decoupled GS manifold)
                    nodes = S.nodes()
                    num_nodes = len(nodes)
                    decoupled1 = np.zeros((num_nodes, num_nodes))
                    decoupled2 = np.zeros((num_nodes, num_nodes))
                    for p, node_index_1 in enumerate(nodes):
                        for q, node_index_2 in enumerate(nodes):
                            decoupled1[p, q] = H1[node_index_1, node_index_2]
                            decoupled2[p, q] = H2[node_index_1, node_index_2]
                    E1 = eigh(H1, eigvals_only = True)[0]
                    E2 = eigh(H2, eigvals_only = True)[0]
                    chi_ab = (E2 - 2. * E1 + E0) / (2 * (h_z ** 2.)) - 0.5 * (chi_aa_matrix[a] + chi_aa_matrix[b])
                    chi_ab_matrix[a, b] = chi_ab
                    chi_ab_matrix[b, a] = chi_ab

                    print('chi_ab ',  a, b, 'completed')
            # adding the diagonal elements
            for c in range(lattice.N):
                chi_ab_matrix[c, c] = chi_aa_matrix[c]

            print('h_x = ', h_x, 'completed')

            chi, order_param = np.sum(np.power(chi_ab_matrix, 2.)), abs(np.sum(order_param_matrix))
            chi_arr[l] = chi
            order_param_arr[l] = order_param

        chi_arr_all.append(chi_arr)
        order_param_all.append(order_param_arr)

    chi_arr_ave = np.mean(chi_arr_all, axis = 0)
    order_param_ave = np.mean(order_param_all, axis = 0)

    return chi_arr_all, order_param_all, chi_arr_ave, order_param_ave

def is_connected(basis, Jij, GS_indices, GS_energy, N):
    '''
    draft function: lacking computational efficiency, but checks if the ground manifold is connected at fourth order perturbation theory
    '''
    # Building block matrices
    ES_1_indices = tfim_matrices.Hamming_set(basis, GS_indices, N, GS_indices)
    ES_2_indices = tfim_matrices.Hamming_set(basis, ES_1_indices, N, GS_indices)

    # Building blocks matrices
    PVP = tfim_matrices.PVP(basis, GS_indices, N)
    PVQ1 = tfim_matrices.PVQ_1(basis, GS_indices, ES_1_indices, N)
    Q1VP = np.transpose(PVQ1)
    Q1VQ1 = tfim_matrices.Q_1VQ_1(basis, ES_1_indices, GS_indices, N)
    Q1VQ2 = tfim_matrices.Q_1VQ_2(basis, ES_2_indices, ES_1_indices, GS_indices, N)
    Q2VQ1 = np.transpose(Q1VQ2)

    # energy_gap_matrix_12 (EGM) denotes 1/(E_0-QH_0Q)^2 from Q1 to Q1
    EGM_12 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, GS_energy, 2)
    EGM_13 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, GS_energy, 3)
    EGM_11 = tfim_matrices.energy_gap(basis, Jij, ES_1_indices, GS_energy, 1)
    EGM_21 = tfim_matrices.energy_gap(basis, Jij, ES_2_indices, GS_energy, 1)

    # Start building Hamiltonians

    H_0 = H_app_0(GS_energy, GS_indices)

    H_app_1 = PVP

    H_app_2 = PVQ1 @ EGM_11 @ Q1VP

    H_app_3 = -0.5 * (PVP @ PVQ1 @ EGM_12 @ Q1VP + np.transpose(
        PVP @ PVQ1 @ EGM_12 @ Q1VP)) + PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_11 @ Q1VP

    H_app_4 = 0.5 * (tfim_matrices.hc(PVQ1 @ EGM_13 @ Q1VP @ PVP @ PVP)) - 0.5 * (
        tfim_matrices.hc(PVQ1 @ EGM_12 @ Q1VP @ PVQ1 @ EGM_11 @ Q1VP)) - 1. * (
                  tfim_matrices.hc(PVQ1 @ EGM_11 @ Q1VQ1 @ EGM_12 @ Q1VP @ PVP)) + 1. * (
                          PVQ1 @ EGM_11 @ Q1VQ2 @ EGM_21 @ Q2VQ1 @ EGM_11 @ Q1VP)

    # construct connectivity matrix
    conn_matrix = H_0 + H_app_1 + H_app_2 + H_app_3 + H_app_4
    adj_matrix = np.zeros(np.shape(conn_matrix))
    for m in range(len(GS_indices)):
        for n in range(len(GS_indices)):
            if conn_matrix[m, n] != 0.:
                adj_matrix[m, n] = 1
    G = nx.from_numpy_matrix(adj_matrix)
    connectivity = nx.is_connected(G)
    print("number of ground states: ", len(GS_indices), 'connected:', connectivity)

    return connectivity