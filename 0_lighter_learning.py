import numpy as np

weights = np.array([0.5, 0.48, -0.7])

alpha = 0.1

streetlights = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1],
                         [1, 0, 1]])

walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

input = streetlights[0]
goal_prediction = walk_vs_stop[0]

for i in range(40):
  error_for_all_lights = 0

  for item in range(len(walk_vs_stop)):
    input = streetlights[item]
    goal_prediction = walk_vs_stop[item]

    prediction = input.dot(weights)

    error = (goal_prediction - prediction)**2
    delta = prediction - goal_prediction
    # Delta vector!
    weights -= alpha * (input * delta)

    print('error^2: %.5f, delta: %.5f, delta-weight:' % (error, delta),
          alpha * (input * delta))
    error_for_all_lights += error

  print('global_sum_error^2 = %.5f' % error_for_all_lights)
