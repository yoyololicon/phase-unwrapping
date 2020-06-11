import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import matplotlib.pyplot as plt
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
from utils import wrap_func
from minimum_cost import minimum_cost_PU_grid

shape = (128,) * 2
height = 13 * np.pi
mean = np.array(shape) * 0.5
cov = np.array([15 ** 2, 15 ** 2, ])
rv = multivariate_normal(mean, cov)

x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
output = rv.pdf(np.dstack((x, y)))
output *= height / output.max()
output[:64, :64] = 0
output += np.random.randn(*output.shape) * 0.6
psi = wrap_func(output)
plt.subplot(211)
plt.imshow(output)
plt.subplot(212)
plt.imshow(psi)
plt.show()

# Load an image as a floating-point grayscale
image = color.rgb2gray(img_as_float(data.chelsea()))
# Scale the image to [0, 4*pi]
image = exposure.rescale_intensity(image, out_range=(0, 4 * np.pi))
# Create a phase-wrapped image in the interval [-pi, pi)
psi = np.angle(np.exp(1j * image))

plt.subplot(211)
plt.imshow(image, cmap='gray')
plt.subplot(212)
plt.imshow(psi, cmap='gray')
plt.show()


plt.subplot(211)
plt.imshow(unwrap_phase(psi), cmap='gray')

plt.subplot(212)
plt.imshow(minimum_cost_PU_grid(psi), cmap='gray')
plt.show()
