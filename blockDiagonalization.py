import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import math
import random

from scipy.sparse import csr_matrix
"""
# Example usage
block_sizes = [6, 2,2,3]  # Each block will be of different size
random_seed = 41  # For reproducibility

H = create_random_block_diag(block_sizes, random_seed)
permutation = list(range(sum(block_sizes)))
np.random.shuffle(permutation)

P = create_permutation_matrix(permutation)
matrix = P@H@P.T
P, block_diag_matrix = block_diagonalize(matrix)
"""

def create_random_block_diag(block_sizes, random_seed=None):
    """
    Creates a random block diagonal matrix.

    Parameters:
    - num_blocks: int, the number of blocks.
    - block_sizes: list of ints, sizes of each block.
    - random_seed: int or None, seed for reproducibility (optional).

    Returns:
    - block_diag_matrix: np.ndarray, the resulting block diagonal matrix.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    num_blocks = len(block_sizes)
    blocks = [np.random.rand(size, size)+1j*np.random.rand(size, size) for size in block_sizes[:num_blocks]]
    block_diag_matrix = block_diag(*blocks)
    return csr_matrix(np.around(block_diag_matrix, 1))


def block_diagonalize(matrix):
    """
    Block diagonalizes the given square matrix.

    Parameters:
    - matrix: csr_matrix, the input square matrix.

    Returns:
    - P: np.ndarray, the permutation matrix used to block diagonalize.
    - block_diag_matrix: np.ndarray, the block diagonalized matrix.
    """
    # Find connected components of the adjacency graph of the matrix
    if isinstance(matrix,csr_matrix)==False:
        matrix = csr_matrix(matrix)
    n_components, labels = connected_components(csgraph=matrix, directed=False, return_labels=True)

    list_of_permutation_blocks = [np.where(labels == comp_id)[0] for comp_id in range(n_components)]

    # returns a list of list of components, like [[1,2],[3]] which we then  flatten via concatenate
    permutation = np.concatenate(list_of_permutation_blocks)
    # Create the permutation matrix P
    P = create_permutation_matrix(permutation)

    # Apply the permutation to block diagonalize the matrix
    block_diag_matrix = P.T @ matrix @ P
    print(block_diag_matrix.toarray())
    # Extract blocks
    blocks = []
    start = 0
    for block_indices in list_of_permutation_blocks:
        b_size = len(block_indices)
        blocks.append(block_diag_matrix[start:start + b_size, start:start + b_size].toarray())
        start += b_size
    for block in blocks:
        print(block)
        norm_value = np.linalg.norm(block)
        print("Norm =",norm_value)
        # Compute the determinant
        det_value = np.linalg.det(block)
        print("Det=",det_value)

    return P, blocks


def create_permutation_matrix(rows):
    """
    Creates a permutation matrix from a permutation vector called rows.

    (1) Rows are given by permutation
    (2) Columns are a simple range
    (3) Data is all ones
    """
    size = len(rows)
    cols = np.arange(size)
    data = np.ones(size, dtype=int)
    P = csr_matrix((data, (rows, cols)), shape=(size, size))
    return P


def create_permutation_projectors(rows):
    """
    Creates a permutation matrix from a permutation vector called rows.

    (1) Rows are given by permutation
    (2) Columns are a simple range
    (3) Data is all ones
    """
    size = len(rows)
    cols = np.arange(size)
    data = np.ones(size, dtype=int)
    P = csr_matrix((data, (rows, cols)), shape=(size, size))
    return P

#old code to block diagonalize


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

    # print("Permuted Matrix:")
    # print(process_complex_list(permuted_matrix))

    return permuted_matrix, sub_blocks

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

