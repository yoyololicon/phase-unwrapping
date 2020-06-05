import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import matplotlib.pyplot as plt


def wrap_func(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


shape = (128,) * 2
height = 24 * np.pi
mean = np.array(shape) * 0.5
cov = np.array([15 ** 2, 15 ** 2, ])
rv = multivariate_normal(mean, cov)

x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
output = rv.pdf(np.dstack((x, y)))
output *= height / output.max()
output[:64, :64] = 0
#output += np.random.randn(*output.shape) * 0.3
psi = wrap_func(output)
plt.subplot(211)
plt.imshow(output)
plt.subplot(212)
plt.imshow(psi)
plt.show()

psi1 = wrap_func(np.diff(psi, axis=0))
psi2 = wrap_func(np.diff(psi, axis=1))

N, M = shape
total_nodes = N * M
s = total_nodes
t = s + 1

max_step = 1
scheduled_steps = list(range(1, max_step + 1)) * 1


def V(x):
    return np.abs(x) ** 0.001
    # return np.where(np.abs(x) > 0.5, 0.25 + np.abs((np.abs(x) - 0.5))** 0.1, x * x)


G = nx.Graph()

demands = np.round(-(psi1[:, 1:] - psi1[:, :-1] - psi2[1:, :] + psi2[:-1, :]) * 0.5 / np.pi).astype(np.int)
index = np.arange(N * M).reshape(N, M)[:-1, :-1].ravel()

for i, d in zip(index, demands.ravel()):
    G.add_node(i, demand=d)

G.add_node(-1, demand=-demands.sum())

for i in range(N - 1):
    G.add_weighted_edges_from(zip(range(i * M, (i + 1) * M),
                                  range((i + 1) * M, (i + 2) * M),
                                  (demands[i] == 0).astype(np.int).tolist() + [0]))

for i in range(M - 1):
    G.add_weighted_edges_from(zip(range(i, N * M, M),
                                  range(i + 1, N * M, M),
                                  (demands[:, i] == 0).astype(np.int).tolist() + [0]))

G.add_edges_from(zip([-1] * M, range(M)))
G.add_edges_from(zip([-1] * M, range((N - 1) * M, N * M)))
G.add_edges_from(zip([-1] * M, range(0, N * M, M)))
G.add_edges_from(zip([-1] * M, range(M - 1, N * M, M)))

G = G.to_directed()
plt.imshow(demands)
plt.show()

cost, flowdict = nx.network_simplex(G)
print(cost)

K2 = np.empty((N, M - 1))
K1 = np.empty((N - 1, M))

for i in range(N - 1):
    for j in range(M):
        if j == 0:
            K1[i][0] = flowdict[i * M + j][-1] - flowdict[-1][i * M]
        else:
            K1[i][j] = flowdict[i * M + j - 1][i * M + j] - flowdict[i * M + j][i * M + j - 1]

for i in range(N):
    for j in range(M - 1):
        if i == 0:
            K2[i][j] = flowdict[-1][j] - flowdict[j][-1]
        else:
            K2[i][j] = flowdict[(i - 1) * M + j][i * M + j] - flowdict[i * M + j][(i - 1) * M + j]

psi[1:, 0] = np.cumsum(psi1[:, 0]) + psi[0, 0]
psi[:, 1:] = np.cumsum(psi2, axis=1) + psi[:, :1]

plt.subplot(311)
plt.imshow(demands)
plt.subplot(312)
plt.imshow(K1)
plt.subplot(313)
plt.imshow(K2)
plt.show()

psi1 += K1 * 2 * np.pi
psi2 += K2 * 2 * np.pi

psi[1:, 0] = np.cumsum(psi1[:, 0]) + psi[0, 0]
psi[:, 1:] = np.cumsum(psi2, axis=1) + psi[:, :1]

plt.imshow(psi)
plt.show()
