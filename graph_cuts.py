import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import scipy_maxflow
import maxflow

def puma_new(x: np.ndarray, max_jump=1, p=1):
    N, M = x.shape

    if max_jump > 1:
        jump_steps = list(range(1, max_jump + 1)) * 2
    else:
        jump_steps = [max_jump]

    total_nodes = N * M
    t = total_nodes
    s = t + 1

    def V(x):
        return np.abs(x) ** p

    index = np.arange(total_nodes).reshape(N, M)
    edges = np.vstack((np.vstack((index[:, :-1].ravel(), index[:, 1:].ravel())).T,
                       np.vstack((index[:-1].ravel(), index[1:].ravel())).T))
    psi = x.flatten()
    K = np.zeros_like(psi)

    def cal_Ek(K, psi, i, j):
        return np.sum(V(2 * np.pi * (K[j] - K[i]) - psi[i] + psi[j]))

    prev_Ek = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

    for step in jump_steps:
        while 1:
            G = maxflow.Graph[float]()
            G.add_grid_nodes(x.shape)

            i, j = edges[:, 0], edges[:, 1]
            psi_diff = psi[i] - psi[j]
            a = 2 * np.pi * (K[j] - K[i]) - psi_diff
            e00 = e11 = V(a)
            e01 = V(a - 2 * np.pi * step)
            e10 = V(a + 2 * np.pi * step)
            weight = np.maximum(0, e10 + e01 - e00 - e11)

            G.add_edges(edges[:, 0], edges[:, 1], weight, np.zeros_like(weight))

            tmp_st_weight = np.zeros((2, total_nodes))

            for i in range(edges.shape[0]):
                u, v = edges[i]
                tmp_st_weight[0, u] += max(0, e10[i] - e00[i])
                tmp_st_weight[0, v] += max(0, e11[i] - e10[i])
                tmp_st_weight[1, u] += -min(0, e10[i] - e00[i])
                tmp_st_weight[1, v] += -min(0, e11[i] - e10[i])

            tmp_st_weight = tmp_st_weight.reshape(2, *x.shape)
            G.add_grid_tedges(index, tmp_st_weight[0], tmp_st_weight[1])

            G.maxflow()
            partition = G.get_grid_segments(index)

            #plt.imshow(~partition, aspect='auto')
            #plt.show()

            partition = partition.ravel()

            K[~partition] += step

            energy = cal_Ek(K, psi, edges[:, 0], edges[:, 1])
            print(energy)
            if energy < prev_Ek:
                prev_Ek = energy
            else:
                K[~partition] -= step
                break

    return (psi + 2 * np.pi * K).reshape(N, M)


def puma(x: np.ndarray, max_jump=1, p=1):
    N, M = x.shape

    if max_jump > 1:
        jump_steps = list(range(1, max_jump + 1)) * 2
    else:
        jump_steps = [max_jump]

    total_nodes = N * M
    t = total_nodes
    s = t + 1

    def V(x):
        return np.abs(x) ** p

    index = np.arange(total_nodes).reshape(N, M)
    edges = np.vstack((np.vstack((index[:, :-1].ravel(), index[:, 1:].ravel())).T,
                       np.vstack((index[:-1].ravel(), index[1:].ravel())).T))
    psi = x.flatten()
    K = np.zeros_like(psi)

    def cal_Ek(K, psi, i, j):
        return np.sum(V(2 * np.pi * (K[j] - K[i]) - psi[i] + psi[j]))

    prev_Ek = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

    for step in jump_steps:
        while 1:
            G = nx.DiGraph()

            i, j = edges[:, 0], edges[:, 1]
            psi_diff = psi[i] - psi[j]
            a = 2 * np.pi * (K[j] - K[i]) - psi_diff
            e00 = e11 = V(a)
            e01 = V(a - 2 * np.pi * step)
            e10 = V(a + 2 * np.pi * step)
            weight = np.maximum(0, e10 + e01 - e00 - e11)
            weight = (1e+2 * weight).astype(int)

            G.add_weighted_edges_from(zip(i, j, weight),
                                      weight='capacity')

            tmp_st_weight = np.zeros((2, total_nodes))

            for i in range(edges.shape[0]):
                u, v = edges[i]
                tmp_st_weight[0, u] += max(0, e10[i] - e00[i])
                tmp_st_weight[0, v] += max(0, e11[i] - e10[i])
                tmp_st_weight[1, u] += -min(0, e10[i] - e00[i])
                tmp_st_weight[1, v] += -min(0, e11[i] - e10[i])

            tmp_st_weight = (1e+2 * tmp_st_weight).astype(int).ravel()
            st_edges = np.array(list(zip([s] * total_nodes + list(range(total_nodes)),
                                         list(range(total_nodes)) + [t] * total_nodes)))

            G.add_weighted_edges_from(zip(st_edges[:, 0],
                                          st_edges[:, 1],
                                          tmp_st_weight),
                                      weight='capacity')

            cut_value, partition = nx.minimum_cut(G, s, t,
                                                  flow_func=scipy_maxflow)
            reachable, _ = partition
            reachable.discard(s)

            K[list(reachable)] += step

            energy = cal_Ek(K, psi, edges[:, 0], edges[:, 1])
            print(energy)

            if energy < prev_Ek:
                prev_Ek = energy
            else:
                K[list(reachable)] -= step
                break

    return (psi + 2 * np.pi * K).reshape(N, M)
