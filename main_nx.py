import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import matplotlib.pyplot as plt


def wrap_func(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


shape = (128,) * 2
height = 13 * np.pi
mean = np.array(shape) * 0.5
cov = np.array([15 ** 2, 15 ** 2, ])
rv = multivariate_normal(mean, cov)

x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
output = rv.pdf(np.dstack((x, y)))
output *= height / output.max()
#output[:64, :64] = 0
output += np.random.randn(*output.shape) * 1.07
psi = wrap_func(output)
plt.subplot(211)
plt.imshow(output)
plt.subplot(212)
plt.imshow(psi)
plt.show()

N, M = shape
total_nodes = N * M
s = total_nodes
t = s + 1

max_step = 1
scheduled_steps = list(range(1, max_step + 1)) * 1


def V(x):
    return np.abs(x) ** 0.001
    #return np.where(np.abs(x) > 0.5, 0.25 + np.abs((np.abs(x) - 0.5))** 0.1, x * x)


G = nx.DiGraph()
G.add_edges_from(zip(range(total_nodes - M), range(M, total_nodes)), capacity=0.)
for i in range(N):
    G.add_edges_from(zip(range(i * M, (i + 1) * M - 1), range(i * M + 1, (i + 1) * M)), capacity=0.)

edges = np.array(G.edges())
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


for step in scheduled_steps:
    while 1:
        print("Setting weights...")
        i, j = edges[:, 0], edges[:, 1]
        psi_diff = psi[i] - psi[j]
        a = 2 * np.pi * (K[j] - K[i]) - psi_diff
        e00 = e11 = V(a)
        e01 = V(a - 2 * np.pi * step)
        e10 = V(a + 2 * np.pi * step)
        weight = np.maximum(0, e10 + e01 - e00 - e11)

        G.add_weighted_edges_from(zip(edges[:, 0], edges[:, 1], weight),
                                  weight='capacity')

        tmp_st_weight = np.zeros((2, total_nodes))

        for i in range(edges.shape[0]):
            u, v = edges[i]
            tmp_st_weight[0, u] += max(0, e10[i] - e00[i])
            tmp_st_weight[0, v] += max(0, e11[i] - e10[i])
            tmp_st_weight[1, u] += max(0, -e10[i] + e00[i])
            tmp_st_weight[1, v] += max(0, - e11[i] + e10[i])

        tmp_st_weight = tmp_st_weight.ravel()
        mask = tmp_st_weight > 0
        st_edges = np.array(list(zip([s] * total_nodes + list(range(total_nodes)),
                                      list(range(total_nodes)) + [t] * total_nodes)))
        st_edges = st_edges[mask]
        tmp_st_weight = tmp_st_weight[mask]
        print(mask.sum())
        G.add_weighted_edges_from(zip(st_edges[:, 0],
                                      st_edges[:, 1],
                                      tmp_st_weight),
                                  weight='capacity')

        print("caculating flow and min cut")
        cut_value, partition = nx.minimum_cut(G, s, t,
                                             flow_func=nx.algorithms.flow.boykov_kolmogorov)
        reachable, _ = partition
        reachable.discard(s)
        print(len(reachable))

        K2 = K.copy()
        K2[list(reachable)] += 1

        plt.imshow((K2 - K).reshape(N, M))
        plt.show()

        energy = cal_Ek(K2, psi, edges[:, 0], edges[:, 1])
        print(energy, step)

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
