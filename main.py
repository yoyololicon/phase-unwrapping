import numpy as np
from scipy.stats import multivariate_normal
import networkx as nx
import graph_tool
from graph_tool.flow import boykov_kolmogorov_max_flow, min_st_cut, push_relabel_max_flow, edmonds_karp_max_flow
import matplotlib.pyplot as plt


def wrap_func(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


shape = (128,) * 2
height = 20 * np.pi
mean = np.array(shape) * 0.5
cov = np.array([15 ** 2, 15 ** 2, ])
rv = multivariate_normal(mean, cov)

x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
output = rv.pdf(np.dstack((x, y)))
output *= height / output.max()
output[:64, :64] = 0
output += np.random.randn(*output.shape) * np.pi * 0.2
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

max_step = 7
scheduled_steps = list(range(1, max_step + 1)) * 2


def V(x):
    return np.abs(x) ** 0.2
    #return np.where(np.abs(x) > 0.5, 0.25 + np.abs((np.abs(x) - 0.5))** 0.1, x * x)


G = graph_tool.Graph()
G.add_edge_list(zip(range(total_nodes - M), range(M, total_nodes)))
for i in range(N):
    G.add_edge_list(zip(range(i * M, (i + 1) * M - 1), range(i * M + 1, (i + 1) * M)))
G.add_edge_list(zip([s] * total_nodes, range(total_nodes)))
G.add_edge_list(zip(range(total_nodes), [t] * total_nodes))

edges = G.get_edges().T
psi = psi.flatten()
K = np.round(np.random.rand(*psi.shape) * 0)
s_vtx, t_vtx = G.vertex(s), G.vertex(t)

Ek = 0
for e in G.edges():
    if e.source() != s_vtx and e.target() != t_vtx:
        i, j = int(e.source()), int(e.target())
        Ek += V(2 * np.pi * (K[j] - K[i]) - psi[i] + psi[j])


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
        cap = G.new_edge_property('double', 0)

        for e in G.edges():
            if e.source() == s_vtx:
                for vhat in G.get_out_neighbors(e.target()):
                    if vhat != t_vtx:
                        e00, _, e10, e11 = get_weights(psi, K, e.target(), vhat, step)
                        cap[e] += max(0, e10 - e00)
                for v in G.get_in_neighbors(e.target()):
                    if v != s_vtx:
                        e00, _, e10, e11 = get_weights(psi, K, v, e.target(), step)
                        cap[e] += max(0, e11 - e10)
            elif e.target() == t_vtx:
                for vhat in G.get_out_neighbors(e.source()):
                    if vhat != t_vtx:
                        e00, _, e10, e11 = get_weights(psi, K, e.source(), vhat, step)
                        cap[e] += max(0, -e10 + e00)
                for v in G.get_in_neighbors(e.source()):
                    if v != s_vtx:
                        e00, _, e10, e11 = get_weights(psi, K, v, e.source(), step)
                        cap[e] += max(0, -e11 + e10)
            else:
                e00, e01, e10, e11 = get_weights(psi, K, e.source(), e.target(), step)
                weight = max(0, e10 + e01 - e00 - e11)
                cap[e] = weight

        res = push_relabel_max_flow(G, s_vtx, t_vtx, cap)
        part = min_st_cut(G, s_vtx, cap, res)

        # cut_value, partition = nx.minimum_cut(G, s, t,
        #                                     flow_func=nx.algorithms.flow.shortest_augmenting_path)
        # reachable, _ = partition
        # reachable.discard(s)
        # print(len(reachable))

        K2 = K.copy()
        count = 0
        for i in range(total_nodes):
            if part[G.vertex(i)]:
                K2[i] = K[i] + step
                count += 1

        plt.imshow((K2 - K).reshape(*shape))
        plt.show()

        energy = 0
        for e in G.edges():
            if e.source() != s_vtx and e.target() != t_vtx:
                i, j = int(e.source()), int(e.target())
                energy += V(2 * np.pi * (K2[j] - K2[i]) - psi[i] + psi[j])
        print(count, energy, step)

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
