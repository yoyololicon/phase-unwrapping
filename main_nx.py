import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from utils import wrap_func
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase


def insarpair_v2(power, cohe, phase, npower):
    M, N = phase.shape
    w1 = 1 / np.sqrt(2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * np.sqrt(power);
    w2 = 1 / np.sqrt(2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * np.sqrt(power);

    n1 = np.sqrt(npower / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    n2 = np.sqrt(npower / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))

    x1 = cohe * w1 + np.sqrt(1 - cohe ** 2) * w2 + n1
    x2 = w1 * np.exp(-1j * phase) + n2
    return x1, x2


def scipy_maxflow(G: nx.DiGraph, s, t, capacity='capacity', **kwargs):
    cap = nx.get_edge_attributes(G, capacity)
    edges = G.edges
    cap = [cap[e] for e in edges]

    edges = np.array(edges)
    sparse_G = sparse.csr_matrix((cap, (edges[:, 0], edges[:, 1])),
                                 shape=(edges.max() + 1, ) * 2)

    res = sparse.csgraph.maximum_flow(sparse_G, s, t)

    R = nx.algorithms.flow.utils.build_residual_network(G, 'capacity')
    R.graph['flow_value'] = res.flow_value
    residual = res.residual

    for i, j in R.edges:
        R[i][j]['flow'] = residual[i, j]

    return R


shape = (100,) * 2
height = 14 * np.pi
mean = np.array(shape) * 0.5
cov = np.array([10 ** 2, 10 ** 2, ])
rv = multivariate_normal(mean, cov)

x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
output = rv.pdf(np.dstack((x, y)))
output *= height / output.max()
output[:50, :50] = 0
# output += np.random.randn(*output.shape) * 0.8
co = 0.8
x1, x2 = insarpair_v2(1, co, output, 0)
psi = np.angle(x1 * np.conj(x2))
#psi = wrap_func(output)

# Load an image as a floating-point grayscale
image = color.rgb2gray(img_as_float(data.chelsea()))
# Scale the image to [0, 4*pi]
image = exposure.rescale_intensity(image, out_range=(0, 4 * np.pi))
# Create a phase-wrapped image in the interval [-pi, pi)
# psi = np.angle(np.exp(1j * image))
# output = image

plt.subplot(211)
plt.imshow(output, cmap='gray')
plt.subplot(212)
plt.imshow(psi, cmap='gray')
plt.show()

N, M = psi.shape
total_nodes = N * M
t = total_nodes
s = t + 1

max_step = 1
scheduled_steps = list(range(1, max_step + 1)) * 1


def V(x):
    return np.abs(x)  ** 0.2
    # return np.abs(np.round(x * 0.5 / np.pi) * 2 * np.pi) ** 2
    # return np.where(np.abs(x) > 0.5, 0.25 + np.abs((np.abs(x) - 0.5))** 0.1, x * x)


index = np.arange(total_nodes).reshape(N, M)
edges = np.vstack((np.vstack((index[:, :-1].ravel(), index[:, 1:].ravel())).T,
                   np.vstack((index[:-1].ravel(), index[1:].ravel())).T))
psi = psi.flatten()
K = np.round(np.random.rand(*psi.shape) * 0)


def cal_Ek(K, psi, i, j):
    return np.sum(V(2 * np.pi * (K[j] - K[i]) - psi[i] + psi[j]))


Ek = cal_Ek(K, psi, edges[:, 0], edges[:, 1])
print(Ek)


def get_weights(psi, K, i, j, s=1):
    i, j = int(i), int(j)
    psi_diff = psi[i] - psi[j]
    a = 2 * np.pi * (K[j] - K[i]) - psi_diff
    e00 = e11 = V(a)
    e01 = V(a - 2 * np.pi * s)
    e10 = V(a + 2 * np.pi * s)
    return e00, e01, e10, e11


G = nx.DiGraph()

while 1:
    print("Setting weights...")
    i, j = edges[:, 0], edges[:, 1]
    psi_diff = psi[i] - psi[j]
    a = 2 * np.pi * (K[j] - K[i]) - psi_diff
    e00 = e11 = V(a)
    e01 = V(a - 2 * np.pi * 1)
    e10 = V(a + 2 * np.pi * 1)
    weight = np.maximum(0, e10 + e01 - e00 - e11)

    G.add_weighted_edges_from(zip(i, j, (200 * weight).astype(int)),
                              weight='capacity')

    tmp_st_weight = np.zeros((2, total_nodes))

    for i in range(edges.shape[0]):
        u, v = edges[i]
        tmp_st_weight[0, u] += max(0, e10[i] - e00[i])
        tmp_st_weight[0, v] += max(0, e11[i] - e10[i])
        tmp_st_weight[1, u] += -min(0, e10[i] - e00[i])
        tmp_st_weight[1, v] += -min(0, e11[i] - e10[i])

    tmp_st_weight = tmp_st_weight.ravel()
    mask = tmp_st_weight > 0
    st_edges = np.array(list(zip([s] * total_nodes + list(range(total_nodes)),
                                 list(range(total_nodes)) + [t] * total_nodes)))
    # st_edges = st_edges[mask]
    # tmp_st_weight = tmp_st_weight[mask]
    # print(mask.sum())
    G.add_weighted_edges_from(zip(st_edges[:, 0],
                                  st_edges[:, 1],
                                  (200 * tmp_st_weight).astype(int)),
                              weight='capacity')

    print("caculating flow and min cut")
    cut_value, partition = nx.minimum_cut(G, s, t,
                                          flow_func=scipy_maxflow)
    reachable, _ = partition
    reachable.discard(s)
    print(len(reachable))

    K2 = K.copy()
    K2[list(reachable)] += 1

    plt.imshow((K2 - K).reshape(N, M))
    plt.show()

    energy = cal_Ek(K2, psi, edges[:, 0], edges[:, 1])
    print(energy)

    if energy < Ek:
        Ek = energy
        K = K2
    else:
        break

    plt.subplot(211)
    plt.imshow(K.reshape(*shape))
    plt.subplot(212)
    plt.imshow((psi + 2 * np.pi * K).reshape(*shape))
    plt.show()
