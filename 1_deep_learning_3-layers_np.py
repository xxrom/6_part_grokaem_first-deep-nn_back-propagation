import numpy as np

np.random.seed(1)


def relu(x):
  # if x > 0 => return x
  # if x <= 0 => return 0
  return (x > 0) * x


alpha = 0.2
hidden_size = 4

streetlights = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]])

walk_vs_stop = np.array([[1, 1, 0, 0]]).T

# np.random.random - [0.0, 1.0]
# 2 * np.random.random - [0.0, 2.0]
# 2 * np.random.random -1 - [-1.0, 1.0]
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1
print(weights_0_1)
print(weights_1_2)

'''
EXAMPLE:

mult_vector_on_maxtrix

(1,3) 1 row, 3 columns
vector = [[1, 2, 3]]
(3,4) 3 row, 4 columns
matrix = [
  [1,1,1,1],
  [2,2,2,2],
  [3,3,3,3],
]

out = [0,0,0,0]
out[1] =
  vector[0]*matrix[0][1] +
  vector[1]*matrix[1][1] +
  vector[2]*matrix[2][1]

'''


def mult_vector_on_maxtrix(vector, matrix):
  output = [0] * len(matrix[0])
  print('out', output)
  for i in range(len(vector)):
    for j in range(len(matrix[0])):
      output[j] += vector[i] * matrix[i][j]

  return output


layers_0 = streetlights[0]
# for i in range(layers_1):
#  for j in range(weights_0_1[0]):
#   layers_1[i] += layers_0[i] * weights_0_1[i][j]
layers_1 = np.dot(layers_0, weights_0_1)
print('list', weights_0_1.tolist())
print('multi', mult_vector_on_maxtrix(list(layers_0), weights_0_1.tolist()))
print('layer_1', layers_1)
layers_2 = np.dot(layers_1, weights_1_2)

print(layers_0)
print(layers_1)
print(layers_2)
