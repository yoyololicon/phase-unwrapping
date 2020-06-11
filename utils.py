import numpy as np


def wrap_func(x):
    return (x + np.pi) % (2 * np.pi) - np.pi