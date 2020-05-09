import numpy as np

weights = np.array([0.5, 0.48, -0.7])

alpha = 0.1

streetlights = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1],
                         [1, 0, 1]])

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

input = streetlights[0]
goal_prediction = walk_vs_stop[0]

for i in range(20):
  prediction = input.dot(weights)

  error = (goal_prediction - prediction)**2
  delta = prediction - goal_prediction
  # Delta vector!
  weights -= alpha * (input * delta)

  print('error^2: %.5f, delta: %.5f, delta-weight:' % (error, delta),
        alpha * (input * delta))
