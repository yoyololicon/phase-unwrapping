import numpy as np
import networkx as nx
from utils import wrap_func as W


def minimum_cost_PU_grid(x: np.ndarray, capacity=None):
    assert x.ndim == 2, "Input x should be a 2d array!"
    N, M = x.shape

    if capacity is None:
        cap_args = dict()
    else:
        cap_args = {'capacity', capacity}

    psi1 = W(np.diff(x, axis=0))
    psi2 = W(np.diff(x, axis=1))

    G = nx.Graph()

    demands = np.round(-(psi1[:, 1:] - psi1[:, :-1] - psi2[1:, :] + psi2[:-1, :]) * 0.5 / np.pi).astype(np.int)
    index = np.arange(N * M).reshape(N, M)[:-1, :-1]

    G.add_nodes_from(zip(index.ravel(), [{'demand': d} for d in demands.ravel()]))
    G.add_node(-1, demand=-demands.sum())

    edges = np.vstack((
        np.vstack((index[:, :-1].ravel(), index[:, 1:].ravel())).T,
        np.vstack((index[:-1].ravel(), index[1:].ravel())).T)
    )
    weights = np.concatenate(
        (((demands[:, :-1] == 0)).ravel(),
         ((demands[:-1] == 0)).ravel())
    ).astype(np.int).tolist()

    G.add_weighted_edges_from(zip(edges[:, 0], edges[:, 1], weights), **cap_args)

    G.add_weighted_edges_from(zip([-1] * (M - 1),
                                  range(M - 1),
                                  (demands[0] == 0).astype(np.int).tolist())
                              , **cap_args)
    G.add_weighted_edges_from(
        zip([-1] * (M - 1),
            range((N - 2) * M, (N - 1) * M - 1),
            (demands[-1] == 0).astype(np.int).tolist()),
        **cap_args)
    G.add_weighted_edges_from(
        zip([-1] * (N - 1),
            range(0, (N - 1) * M, M),
            (demands[:, 0] == 0).astype(np.int).tolist()),
        **cap_args)
    G.add_weighted_edges_from(
        zip([-1] * (N - 1),
            range(M - 2, (N - 1) * M, M),
            (demands[:, -1] == 0).astype(np.int).tolist()),
        **cap_args)

    G = G.to_directed()

    cost, flowdict = nx.network_simplex(G)

    K2 = np.empty((N, M - 1))
    K1 = np.empty((N - 1, M))

    for i in range(N - 1):
        for j in range(M):
            if j == 0:
                K1[i][0] = flowdict[-1][i * M] - flowdict[i * M][-1]
            elif j == M - 1:
                K1[i][M - 1] = flowdict[i * M + j - 1][-1] - flowdict[-1][i * M + j - 1]
            else:
                K1[i][j] = flowdict[i * M + j - 1][i * M + j] - flowdict[i * M + j][i * M + j - 1]

    for i in range(N):
        for j in range(M - 1):
            if i == 0:
                K2[i][j] = flowdict[-1][j] - flowdict[j][-1]
            elif i == N - 1:
                K2[i][j] = flowdict[(i - 1) * M + j][-1] - flowdict[-1][(i - 1) * M + j]
            else:
                K2[i][j] = flowdict[(i - 1) * M + j][i * M + j] - flowdict[i * M + j][(i - 1) * M + j]

    K2[0, 0] = K2[0, -1] = K2[-1, 0] = K2[-1, -1] = 0

    psi1 += K1 * 2 * np.pi
    psi2 += K2 * 2 * np.pi

    y = np.full_like(x, x[0, 0])
    y[1:, 0] = np.cumsum(psi1[:, 0])
    y[:, 1:] = np.cumsum(psi2, axis=1) + y[:, :1]
    return y
