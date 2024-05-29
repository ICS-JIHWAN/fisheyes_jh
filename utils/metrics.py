import numpy as np


def gaussuian_filter(kernel_size, sigma=1, mu=0):
    # Generating 2D grids 'x' and 'y' using meshgrid with 10 evenly spaced points from -1 to 1
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))

    # Calculating the Euclidean distance 'd' from the origin using the generated grids 'x' and 'y'
    d = np.sqrt(x * x + y * y)

    # Calculating the Gaussian-like distribution 'g' based on the distance 'd', sigma, and mu
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    return g
