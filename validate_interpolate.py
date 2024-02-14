from scipy.interpolate import RegularGridInterpolator
import numpy as np


x = np.linspace(0,3, 20 )
y = np.linspace(0, 1, 20)  # Example y-values

x_pre = np.zeros(10)  # 50 elements with the same x-coordinate
y_pre = np.zeros(10)

x_after = np.ones(10)
y_after = np.ones(10) 


x = np.concatenate((x_pre, x))
y = np.concatenate((y_pre, y))

x = np.concatenate((x, x_after))
y = np.concatenate((y, y_after))
# Create RegularGridInterpolator
interp = RegularGridInterpolator((np.arange(len(x)),), np.array([x, y]).T, method="linear")

# Define the grid for interpolation
grid_x = np.linspace(0, len(x) - 1, 100)

# Interpolate values
interpolated_path = interp(grid_x)

# Separate x and y coordinates of the interpolated path
interpolated_x, interpolated_y = interpolated_path.T

# Plotting
import matplotlib.pyplot as plt

plt.plot(x, y, 'o', label='Original Points')
plt.plot(interpolated_x, interpolated_y, label='Interpolated Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

print("finalx", interpolated_x)
print("finaly", interpolated_y)