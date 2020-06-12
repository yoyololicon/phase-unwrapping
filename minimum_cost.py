import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from utils import wrap_func as W
import matplotlib.pyplot as plt


def minimum_cost_PU_grid(x: np.ndarray, capacity=None):
    assert x.ndim == 2, "Input x should be a 2d array!"
    N, M = x.shape

    if capacity is None:
        cap_args = dict()
    else:
        cap_args = {'capacity': capacity}

    psi1 = W(np.diff(x, axis=0))
    psi2 = W(np.diff(x, axis=1))

    G = nx.Graph()

    demands = np.round(-(psi1[:, 1:] - psi1[:, :-1] - psi2[1:, :] + psi2[:-1, :]) * 0.5 / np.pi).astype(np.int)
    index = np.arange(N * M).reshape(N, M)
    demands = np.pad(demands, ((0, 1),) * 2, 'constant', constant_values=0)

    G.add_nodes_from(zip(index.ravel(), [{'demand': d} for d in demands.ravel()]))
    G.add_node(-1, demand=-demands.sum())

    edges = np.vstack((
        np.vstack((index[:, :-1].ravel(), index[:, 1:].ravel())).T,
        np.vstack((index[:-1].ravel(), index[1:].ravel())).T)
    )
    weights = np.concatenate(((demands[:, :-1] == 0).ravel(), (demands[:-1] == 0).ravel()))
    print(weights.sum())

    G.add_weighted_edges_from(zip(edges[:, 0], edges[:, 1], weights), **cap_args)

    G.add_edges_from(zip([-1] * M, range(M))
                     , **cap_args)
    G.add_edges_from(
        zip([-1] * M,
            range((N - 1) * M, (N - 0) * M - 1)),
        **cap_args)
    G.add_edges_from(
        zip([-1] * (N - 0),
            range(0, (N - 0) * M, M)),
        **cap_args)
    G.add_edges_from(
        zip([-1] * (N - 0),
            range(M - 1, (N - 0) * M, M), ),
        **cap_args)

    G = G.to_directed()

    cost, flowdict = nx.network_simplex(G)
    print(cost)

    K2 = np.empty((N, M - 1))
    K1 = np.empty((N - 1, M))

    for i in range(N - 1):
        for j in range(M):
            if j == 0:
                K1[i][0] = -flowdict[-1][i * M] + flowdict[i * M][-1]
            else:
                K1[i][j] = -flowdict[i * M + j - 1][i * M + j] + flowdict[i * M + j][i * M + j - 1]

    for i in range(N):
        for j in range(M - 1):
            if i == 0:
                K2[i][j] = flowdict[-1][j] - flowdict[j][-1]
            else:
                K2[i][j] = flowdict[(i - 1) * M + j][i * M + j] - flowdict[i * M + j][(i - 1) * M + j]

    K2[0, 0] = 0

    psi1 += K1 * 2 * np.pi
    psi2 += K2 * 2 * np.pi

    y = np.full_like(x, x[0, 0])
    y[1:, 0] = np.cumsum(psi1[:, 0])
    y[:, 1:] = np.cumsum(psi2, axis=1) + y[:, :1]
    return y


def minimum_cost_PU_sparse(x, y, psi, capacity=None):
    points = np.vstack((x, y)).T
    num_points = points.shape[0]
    tri = Delaunay(points)
    simplex = tri.simplices
    simplex_neighbors = tri.neighbors
    num_simplex = simplex.shape[0]

    if capacity is None:
        cap_args = dict()
    else:
        cap_args = {'capacity': capacity}

    psi_diff = W(psi[simplex] - psi[np.roll(simplex, 1, 1)])
    edges = np.stack((np.roll(simplex, 1, 1), simplex), axis=2).reshape(-1, 2)
    simplex_edges = np.stack((
        np.broadcast_to(np.arange(num_simplex)[:, None], simplex_neighbors.shape),
        np.roll(simplex_neighbors, -1, 1)), axis=2
    ).reshape(-1, 2)
    edges, unique_idx = np.unique(edges, axis=0, return_index=True)
    simplex_edges = simplex_edges[unique_idx]

    demands = np.round(-psi_diff.sum(1) * 0.5 / np.pi).astype(np.int)
    psi_diff = psi_diff.flatten()[unique_idx]

    G = nx.Graph()

    traverse_G = nx.Graph()
    traverse_G.add_edges_from(edges)
    traverse_path = list(nx.algorithms.traversal.breadth_first_search.bfs_edges(traverse_G, 0))

    G.add_nodes_from(zip(range(num_simplex),
                         [{'demand': d} for d in demands]))
    G.add_node(-1, demand=-demands.sum())
    print(demands.sum())

    weights = np.sum(demands[simplex_edges] == 0, 1)
    G.add_weighted_edges_from(zip(simplex_edges[:, 0], simplex_edges[:, 1], weights), **cap_args)

    G = G.to_directed()
    cost, flowdict = nx.network_simplex(G)
    print(cost)

    K = np.zeros(edges.shape[0])

    for j in range(simplex_edges.shape[0]):
        u, v = simplex_edges[j]
        K[j] = -flowdict[u][v] + flowdict[v][u]

    psi_diff += K * 2 * np.pi
    print(K.max(), K.min(), np.abs(K).sum())

    psi_dict = {i: {} for i in range(num_points)}

    for diff, (u, v) in zip(psi_diff, edges):
        psi_dict[u][v] = diff
        psi_dict[v][u] = -diff

    result = psi.copy()
    for u, v in traverse_path:
        result[v] = result[u] + psi_dict[u][v]
    return result