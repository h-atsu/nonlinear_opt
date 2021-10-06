import numpy as np
import matplotlib.pyplot as plt
from steepest_gradient_descent import Optimizer
from mpl_toolkits.mplot3d import Axes3D

n_size = 3
A = np.random.random((n_size, n_size))
A = A.T @ A
x0 = np.random.random(n_size)
print(x0)

optimizer = Optimizer(lambda x: x @ A @ x, lambda x: A @ x)
optimizer.optimize(x0)
opt_val = optimizer.opt_val
opt_sol = optimizer.opt_sol
print(opt_val, opt_sol)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
path = optimizer.path
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.plot(path[:, 0], path[:, 1], path[:, 2])
plt.show()
