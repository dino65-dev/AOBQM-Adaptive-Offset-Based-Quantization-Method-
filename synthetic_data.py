import numpy as np
# Synthetic data generation
x = np.linspace(0, 200, 10000)
y = np.linspace(0, 200, 10000)
xv, yv = np.meshgrid(x, y)
sine_wave = np.sin(xv) * np.cos(yv)
noise = np.random.normal(loc=0, scale=0.5, size=(10000, 10000))
float_array = sine_wave + noise
print(float_array)
