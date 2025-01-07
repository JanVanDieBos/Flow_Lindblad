import numpy as np
from scipy.linalg import eigvals

# Define Pauli matrices and the identity matrix
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_0 = np.eye(2)
Sigma = [np.eye(2),np.array([[0, 1], [1, 0]]),np.array([[0, -1j], [1j, 0]]),np.array([[1, 0], [0, -1]])]
def a(N):
    temp = np.zeros((N,N))
    for i in range(N-1): temp[i,i+1] = (i+1)**.5
    return temp
def X(N):
    return 2**.5*(a(N)+a(N).T)
def P(N):
    return 2**.5*(a(N)-a(N).T)/1j
def n(N):
    temp = np.zeros((N,N))
    for i in range(N): temp[i,i] = i
    return temp

from numflow import anticommutator, commutator

def build_Ll(H,L_list):
    dim = H.shape[0]
    I = np.eye(dim)
    Com = np.kron(I, H) - np.kron(H.T, I) #[H,.]
    Unitor = -1j * Com
    L_matrix = -1j * Com
    # Add Lindblad dissipators
    for l in L_list:
        l_dagger = l.conj().T
        l_dagger_l = l_dagger@l #np.dot(l_dagger, l)
        double_l = np.kron(l.conj(), l)
        anticom = np.kron(I, l_dagger_l) + np.kron(l_dagger_l.T, I)
        L_matrix += double_l - 0.5 * anticom
    return L_matrix

def Llrho(H,L_list,rho):
    L_rho = -1j*commutator(H,rho)
    # Add Lindblad dissipators
    for l in L_list:
        l_dagger = l.conj().T;l_dagger_l = l_dagger@l
        L_rho += l@rho@l_dagger-0.5*anticommutator(l_dagger_l,rho)
    return L_rho

def Ll_from_params(j_r,h_vec = np.random.random(3),gamma_vec= np.random.random(4)): #h_vec = [1,0.1,0.1]; gamma_vec
    if isinstance(gamma_vec, list)==False:
        if gamma_vec ==0:
            gamma_vec = [0,0,0,0]
        else:
            gamma_vec = gamma_vec*np.random.random(4)
    H0 = np.sum([h_vec[i]*Sigma[i+1] for i in range(3)], axis=0) #choosing H0 = sigma_z leads to a degeneracy meaning we do not diagonalize the Hamiltonian
    M_r = (np.random.random((2,2))+1j*np.random.random((2,2)));
    M_r = np.linalg.norm(Sigma[1])*M_r/np.linalg.norm(M_r)
    L_list = [np.sum([gamma_vec[i]*Sigma[i] for i in range(4)], axis=0),j_r*M_r]
    L0 = build_Ll(H0,L_list); L0 = L0#/LA.norm(L0)
    unique_d, counts_d = np.unique(np.around(np.diag(L0),9), return_counts=True)
    unique_0, counts_0 = np.unique(np.around(eigvals(L0),9), return_counts=True)
    if 2 in counts_d: print("L_d(0) 2-fold degenerate")
    if 2 in counts_0: print("L 2-fold degenerate")
    return L0
def Ll_bosons(gamma_vec,j_r,N, omega=1): #h_vec = [1,0.1,0.1]; gamma_vec
    H0 = omega*n(N) #choosing H0 = sigma_z leads to a degeneracy meaning we do not diagonalize the Hamiltonian
    M_r = (np.random.random((N,N))+1j*np.random.random((N,N)));
    M_r = np.linalg.norm(n(N))*M_r/np.linalg.norm(M_r)
    L_list = [gamma_vec[0]*X(N)+gamma_vec[1]*P(N),j_r*M_r]
    L0 = build_Ll(H0,L_list); L0 = L0/np.linalg.norm(L0)
    unique_d, counts_d = np.unique(np.around(np.diag(L0),9), return_counts=True)
    degen_d = [x for x in counts_d if x != 1]
    unique_0, counts_0 = np.unique(np.around(eigvals(L0),9), return_counts=True)
    degen_0 = [x for x in counts_0 if x != 1]
    string_note = str(N)+" Bosons. "+"gamma =" +str(gamma_vec)+", j_r="+str(j_r)+". "
    if len(degen_d)!=0: string_note +="L_d(0) degenacy:"+str(degen_d)+". "
    if len(degen_0)!=0: string_note +="L degeneracy:"+str(degen_0)+". "
    return L0, string_note