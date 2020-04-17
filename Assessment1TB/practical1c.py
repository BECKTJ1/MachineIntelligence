import matplotlib

import matplotlib.pyplot as plt
plt.plot([1, 2, 4, 9, 5, 3])
plt.show()

plt.plot([-3, -2, 5, 0], [1, 6, 4, 3])
plt.show()

import numpy as np
x = np.linspace(-2, 2, 500)
y = x**2

plt.plot(x, y)
plt.show()