import numpy as np
from scipy.linalg import expm, eigvals
from numpy.linalg import eigvalsh, inv, norm
import time
from scipy.optimize import linear_sum_assignment
from blockDiagonalization import block_diagonalize


def min_sum_permutation(v, w):
    n = len(v)
    # Create an n x n cost matrix
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.abs(v[i] - w[j])
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate the minimal sum
    min_sum = cost_matrix[row_ind, col_ind].sum()
    
    return min_sum, col_ind

def commutator(A,B): return A@B - B@A

def anticommutator(A,B): return A@B+B@A

def magnus_S(L, dl,L_intR):
    dim = L.shape[0]
    I = np.eye(dim)
    ηl = commutator(L.conj().T,L_intR)
    # Magnus expansion to second order
    dL = commutator(ηl,L);
    dL_intR = dL-dL*I;
    dη = commutator(dL.conj().T,L_intR) + commutator(L.conj().T,dL_intR)
    ddL = commutator(dη,L) + commutator(ηl,dL);
    ddη = commutator(ddL.conj().T,L_intR) + 2*commutator(dL.conj().T,dL_intR) + commutator(L.conj().T,ddL-ddL*I)
    ddζ = (2*ddη - commutator(ηl,dη))/6; 
    zeta = ηl + dη*dl/2+ dl**2*ddζ/2
    Ohm = zeta*dl
    O2 = Ohm@Ohm
    dS = np.matmul(12*I + 6*Ohm + O2, inv(12*I - 6*Ohm + O2) )  #exp_pade S(l+dl,l) Savits eq 31
    return dS

def evolve(L_l,G,I,dl):
    L_intR = -G+L_l-L_l*I
    dS = magnus_S(L_l, dl,L_intR); dSinv_l =inv(dS)
    L_l = dS @ L_l @ dSinv_l
    return L_l

def gershgorin_circles(matrix):
    n = matrix.shape[0]
    centers = [matrix[i, i] for i in range(n)]
    radii = [np.sum(np.abs(matrix[i, :])) - np.abs(centers[i]) for i in range(n)]
    return centers, radii

def evolve_operator(L0, dl=1e-2, g=0):
    min_dl = 1e-6
    max_dl = 0.5
    tol = 1e-4
    L_l = np.copy(L0)
    dim = L_l.shape[0]
    I = np.eye(dim)
    #if g = 0, then we do not add any randomness
    G = np.diag(np.random.rand(dim));
    G = g * G * norm(L_l * I) / norm(G)
    # CC = [[] for i in range(dim)]; RR = [[] for i in range(dim)]
    Rsum = [];
    RsumN = []
    start_time = time.time()  # Record the start time
    ll = [0]
    n_step = 0
    counter = 0
    while True:
        centers, radii = gershgorin_circles(L_l);
        Rsum.append(sum(radii));
        RsumN.append(Rsum[n_step] / Rsum[0])
        # for i in range(dim): RR[i].append(radii[i]); CC[i].append(centers[i])
        if RsumN[n_step] <= .014:
            # Now check if the diagonal is close to what it should be
            eig0, eigd = eigvals(L0), eigvals(L_l * I)
            min_sum, op_perm = min_sum_permutation(eig0, eigd)
            eigd_op = [eigd[op_perm[k]] for k in range(dim)]
            rel_error = [
                abs(eig0[k] - eigd_op[k]) / np.max([abs(eig0[k]), abs(eigd_op[k]), 1e-1 * abs(sum(eig0)) / dim]) for k
                in range(dim)]
            rel_below = [bool(r < 0.01) for r in rel_error]
            if False not in rel_below:
                break
            else:
                counter += 1
            if counter > 5:
                print("eig0", np.around(eig0, 7))
                print("eigd_op", np.around(eigd_op), 7)
                break

        L_temp = np.copy(L_l)
        L_l = evolve(evolve(L_l, G, I, dl / 2), G, I, dl / 2)
        dL = norm(L_l - L_temp)
        # one bigger step
        L_big = evolve(L_temp, G, I, dl)
        error = norm(L_l - L_big)
        if n_step > 40 and sum(np.absolute(np.gradient(RsumN)[n_step - 10:n_step])) <= 1e-14:
            permuted_matrix, disconnected_subblocks = block_diagonalize(np.around(L_l, 10))
            # print("\nSub-Blocks:")
            print("The flow is stuck due to " + str(len(disconnected_subblocks)) + "disconnected subblocks");
            break
        elif error == 0 and dL > 1e-15:
            dl = max_dl
        elif error < tol and dL > 1e-15:
            dl = min(max_dl, dl * (tol / error) ** (1 / 3))
        else:
            dl = max(min_dl, dl * (tol / error) ** (1 / 3))
        # To do: add check if the radius is increasing, then we should not increase the step-size, but then perhapse both are bad
        if time.time() - start_time > dim * 30:
            print("dL =", dL)
            print("dl =", dl)
            print("error =", error)
            print("Breaking loop after " + str(dim * 30) + " seconds");
            break
        ll.append(ll[n_step] + dl)
        n_step += 1

    return L_l, ll, RsumN  # CC, RR





