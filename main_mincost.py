import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import matplotlib.pyplot as plt
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
from utils import wrap_func
from minimum_cost import minimum_cost_PU_grid, minimum_cost_PU_sparse
from graph_cuts import puma, puma_new

shape = (128,) * 2
height = 22 * np.pi
mean = np.array(shape) * 0.5
cov = np.array([15 ** 2, 25 ** 2, ])
rv = multivariate_normal(mean, cov)

x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
output = rv.pdf(np.dstack((x, y)))
output *= height / output.max()
#output[:64, :64] = 0
output += np.random.randn(*output.shape) * 0.8
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
#psi = np.angle(np.exp(1j * image))
image = output

x, y = np.meshgrid(np.arange(psi.shape[1]), np.arange(psi.shape[0]))
sparsity = 0.4
tmp = np.vstack((x.ravel(), y.ravel(), psi.ravel(), image.ravel())).T
np.random.shuffle(tmp)
sp_x, sp_y, sp_psi, sp_image = tmp[:int(tmp.shape[0] * sparsity)].T

plt.subplot(211)
#plt.imshow(image, cmap='gray')
plt.scatter(sp_x, -sp_y, c=sp_image, cmap='gray', s=0.05)
plt.subplot(212)
#plt.imshow(psi, cmap='gray')
plt.scatter(sp_x, -sp_y, c=sp_psi, cmap='gray', s=0.05)
plt.show()


fig = plt.figure(figsize=(8, 12))
plt.subplot(311)
base = unwrap_phase(psi)
plt.imshow(image, cmap='gray')
#plt.scatter(sp_x, -sp_y, c=sp_image, cmap='gray', s=0.05)
#plt.xlim(0, sp_x.max() - 1)
#plt.ylim(-sp_y.max() + 1, 0)
plt.subplot(312)
#plt.scatter(sp_x, -sp_y, c=sp_psi, cmap='gray', s=0.05)
#plt.xlim(0, sp_x.max() - 1)
#plt.ylim(-sp_y.max() + 1, 0)
result = puma(psi, max_jump=1, p=1)
plt.imshow(result, cmap='gray')
plt.subplot(313)
plt.imshow(result - image, cmap='gray')
#plt.scatter(sp_x, -sp_y, c=minimum_cost_PU_sparse(sp_x, sp_y, sp_psi, capacity=5), cmap='gray', s=0.05)
#plt.xlim(0, sp_x.max() - 1)
#plt.ylim(-sp_y.max() + 1, 0)
plt.show()
