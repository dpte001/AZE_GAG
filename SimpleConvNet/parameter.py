import numpy as np

data = np.zeros((1,48))
for i in range(0, 48):
    matrix[0][i] = i+1
matrix = matrix.reshape(1,3, 4, 4)
print(matrix)

W = np.zeros(())