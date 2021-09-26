# Name: dispPCAIm.py
# accepts principal components, original file, and shapefile as inputs
# creates raster of principal components in cropped masked region
# specified by shapefile. band order as specified.

import numpy as np

# Accept square matrix A and compatible vector y
# Return solution to Ux = y
def backSubSolve(A, y):
    if len(A) != len(A[0]):  # Ensure A is square
        raise Exception("Matrix must be square.")

    if len(A[0]) != len(y):  # Check compatibility with vector
        raise Exception("Vector must be dimensionally consistent with matrix")

    x = np.empty_like(y, dtype=np.float)  # Create vector y like b for backward sub use

    for j in range(y.size - 1, -1, -1):  # Use lab handout backward substitution code
        x[j] = (y[j] - sum(A[j, (j + 1):y.size] * x[(j + 1):y.size])) / A[j, j]

    return x


# Accept a matrix A and vector b on which to perform householder triangularization
# Return the modified matrix A, modified vector b, and generated R matrix and c1 vector
def houseTriang(A, b):
    
    A = A.astype(float)  # Ensure that the matrix A is a float
    b = b.astype(float)  # Ensure that the vector b is a float

    col = np.size(A, 1)  # Number of columns in matrix
    row = np.size(A, 0)  # Number of rows in matrix

    for k in range(0, col):  # For all the matrix columns

        e = np.zeros(row - k)  # Construct e with length that decreases with each iteration
        e[0] = 1  # Set first element to be one

        x = A[k:, k]  # Grab a column of the matrix A

        vk = np.sqrt(sum(x ** 2)) * e + x  # Construct vector vk to annihilate all subdiagonal column entries
        vk = vk / np.sqrt(sum(vk ** 2))  # Set vk to be itself divided by its two-norm
        vkt = vk.transpose()  # Find the transpose of vk

        vktAprod = np.matmul(vkt, A[k:, k:])  # Find the product of vkt and submatrix of A
        vk.shape = (np.size(vk), 1)  # Ensure that vk is a column vector

        vktAprod.shape = (1, np.size(vktAprod))  # Ensure that prod is a row vector

        aMat = -2 * np.matmul(vk, vktAprod)  # Find the annihilation matrix

        A[k:, k:] = A[k:, k:] + aMat  # Add aniihilation matrix to find evolution of A into R

        vktAProduct = np.array([np.matmul(vkt, b[k:])])

        b[k:] = b[k:] - 2 * np.matmul(vk, vktAProduct)  # Find evolution of vector b into Qtb = [c1, c2]

        R = A[:col, :]  # Extract R matrix from A
        R.shape = (col, col)  # Ensure shape is correct for use in backSubSolve
        c1 = b[:col]  # Extract c1 matrix from b
        c1.shape = (col,)  # Ensure shape is correct for use in backSubSolve

        x = backSubSolve(R, c1)  # Use backwards substitution to find x for R and c1 in Rx=c1

    return A, b, R, c1, x  # Return the matrix A which is now the matrix R and vector b which is now [c1, c2]
