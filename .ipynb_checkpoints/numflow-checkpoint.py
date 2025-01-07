import numpy as np
from scipy.linalg import expm, eigvals
from numpy.linalg import eigvalsh, inv, norm
import time
from scipy.optimize import linear_sum_assignment
import math

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
    min_dl = 1e-6;
    max_dl = 0.5;
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
            permuted_matrix, disconnected_subblocks = permute_to_block_diagonal(np.around(L_l, 10))
            # print("Permuted Matrix:")
            # print(process_complex_list(permuted_matrix))
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

def complex_round_to_first_nonzero(complex_value):
    """
    Applies rounding to first non-zero decimal to both real and imaginary parts of a complex number.
    Returns a rounded complex number object.
    """

    def round_to_first_nonzero(value, decimal_places=10):
        """
        Rounds a number to the first significant non-zero decimal place.
        """
        if value == 0:
            return 0.0
        elif value < 0:
            negative = True
            value = -value
        else:
            negative = False

        shift = 10 ** (int(-math.log10(value)) + 1)
        rounded_value = math.floor(value * shift) / shift

        if negative:
            rounded_value = -rounded_value

        # Format to remove unnecessary zeros
        return float(f"{rounded_value:.{decimal_places}f}")

    real_part = round_to_first_nonzero(complex_value.real)
    imag_part = round_to_first_nonzero(complex_value.imag)
    c = real_part
    return c

def process_complex_list(M):
    """
    Takes a list of complex numbers, applies rounding, and returns a list of rounded complex numbers.
    """
    M_temp = np.copy(M)
    for i in range(len(M)):
        for j in range(len(M)):
            M_temp[i,j] = complex_round_to_first_nonzero(M[i,j])
    return M_temp

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def permute_to_block_diagonal(matrix):
    """
    Reorders a matrix into a block diagonal form by finding a permutation matrix.
    
    Parameters:
    matrix (np.ndarray): The matrix to reorder.
    
    Returns:
    np.ndarray: The permuted matrix that reveals the block diagonal structure.
    list of np.ndarray: The extracted sub-blocks from the permuted matrix.
    """
    # Ensure the matrix is in CSR format for efficient processing
    sparse_matrix = csr_matrix(matrix)
    
    # Find connected components
    n_components, labels = connected_components(csgraph=sparse_matrix, directed=False, return_labels=True)
    
    # Create a permutation index array from the labels
    perm_index = np.argsort(labels)
    
    # Apply permutation to rows and columns
    permuted_matrix = matrix[perm_index, :][:, perm_index]
    
    # Extract sub-blocks based on connected components
    sub_blocks = []
    for i in range(n_components):
        # Extract rows belonging to each component in the permuted matrix
        component_mask = labels[perm_index] == i
        component_rows = np.where(component_mask)[0]
        sub_block = permuted_matrix[np.ix_(component_rows, component_rows)]
        sub_blocks.append(sub_block)
    
    return permuted_matrix, sub_blocks
