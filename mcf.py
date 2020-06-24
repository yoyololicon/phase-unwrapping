import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from utils import wrap_func as W


def mcf(x: np.ndarray, capacity=None):
    assert x.ndim == 2, "Input x should be a 2d array!"

    # construct index for each node
    N, M = x.shape
    index = np.arange(N * M).reshape(N, M)

    if capacity is None:
        cap_args = dict()
    else:
        cap_args = {'capacity': capacity}

    # get pseudo estimation of gradients along x and y axis
    psi1 = W(np.diff(x, axis=0))
    psi2 = W(np.diff(x, axis=1))

    G = nx.Graph()

    demands = np.round(-(psi1[:, 1:] - psi1[:, :-1] - psi2[1:, :] + psi2[:-1, :]) * 0.5 / np.pi).astype(np.int)
    # for convenience let's pad the demands so it match the shape of image
    # this add N + M - 1 dummy nodes with 0 demand
    demands = np.pad(demands, ((0, 1),) * 2, 'constant', constant_values=0)
    G.add_nodes_from(zip(index.ravel(), [{'demand': d} for d in demands.ravel()]))

    # set earth node index to -1, and its demand is the negative of the sum of all demands,
    # so the total demands is zero
    G.add_node(-1, demand=-demands.sum())

    # edges along x and y axis
    edges = np.vstack((
        np.vstack((index[:, :-1].ravel(), index[:, 1:].ravel())).T,
        np.vstack((index[:-1].ravel(), index[1:].ravel())).T)
    )

    # set the edge weight to 1 when its left (upper) node demands is equal to zero
    # I found it achieve very stable result
    weights = np.concatenate(((demands[:, :-1] == 0).ravel(), (demands[:-1] == 0).ravel()))
    G.add_weighted_edges_from(zip(edges[:, 0], edges[:, 1], weights), **cap_args)

    # add the remaining edges that connected to earth node
    G.add_edges_from(zip([-1] * M, range(M)), **cap_args)
    G.add_edges_from(zip([-1] * M, range((N - 1) * M, N * M - 1)), **cap_args)
    G.add_edges_from(zip([-1] * N, range(0, N * M, M)), **cap_args)
    G.add_edges_from(zip([-1] * N, range(M - 1, N * M, M)), **cap_args)

    # make graph to directed graph, so we can distinguish positive and negative flow
    G = G.to_directed()

    # perform MCF
    cost, flowdict = nx.network_simplex(G)

    # construct K matrix with the same shape as the gradients
    K2 = np.empty((N, M - 1))
    K1 = np.empty((N - 1, M))

    # add the flow to their orthogonal edge
    # the sign of the flow depends on those 4 vectors direction (clockwise or counter-clockwise)
    # when calculating the demands
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

    # the boundary node with index = 0 have only one edge to earth node,
    # so set one of its edge's K to zero
    K2[0, 0] = 0

    # derive correct gradients
    psi1 += K1 * 2 * np.pi
    psi2 += K2 * 2 * np.pi

    # integrate the gradients
    y = np.full_like(x, x[0, 0])
    y[1:, 0] = np.cumsum(psi1[:, 0])
    y[:, 1:] = np.cumsum(psi2, axis=1) + y[:, :1]
    return y


def mcf_sparse(x, y, psi, capacity=None):
    points = np.vstack((x, y)).T
    num_points = points.shape[0]

    # Delaunay triangularization
    tri = Delaunay(points)
    simplex = tri.simplices
    simplex_neighbors = tri.neighbors
    num_simplex = simplex.shape[0]

    if capacity is None:
        cap_args = dict()
    else:
        cap_args = {'capacity': capacity}

    # get pseudo estimation of gradients and vertex edges
    psi_diff = W(psi[simplex] - psi[np.roll(simplex, 1, 1)])
    edges = np.stack((np.roll(simplex, 1, 1), simplex), axis=2).reshape(-1, 2)

    # get corresponding simplex's edges orthogonal to original vertex edges
    simplex_edges = np.stack((
        np.broadcast_to(np.arange(num_simplex)[:, None], simplex_neighbors.shape),
        np.roll(simplex_neighbors, -1, 1)), axis=2
    ).reshape(-1, 2)

    # remove duplicated vertex edges
    edges, unique_idx = np.unique(edges, axis=0, return_index=True)
    simplex_edges = simplex_edges[unique_idx]

    # get demands
    demands = np.round(-psi_diff.sum(1) * 0.5 / np.pi).astype(np.int)
    psi_diff = psi_diff.flatten()[unique_idx]

    G = nx.Graph()
    G.add_nodes_from(zip(range(num_simplex),
                         [{'demand': d} for d in demands]))

    # choose an integration path, here let's use BFS
    traverse_G = nx.Graph()
    traverse_G.add_edges_from(edges)
    traverse_path = list(nx.algorithms.traversal.breadth_first_search.bfs_edges(traverse_G, 0))

    # set earth node index to -1, and its demand is the negative of the sum of all demands,
    # so the total demands is zero
    G.add_node(-1, demand=-demands.sum())

    # set the edge weight to 1 whenever one of its nodes has zero demand
    demands_dummy = np.pad(demands, (0, 1), 'constant', constant_values=0)
    weights = np.any(demands_dummy[simplex_edges] == 0, 1)
    G.add_weighted_edges_from(zip(simplex_edges[:, 0], simplex_edges[:, 1], weights), **cap_args)

    # make graph to directed graph, so we can distinguish positive and negative flow
    G = G.to_directed()

    # perform MCF
    cost, flowdict = nx.network_simplex(G)

    # construct K matrix with the same shape as the edges
    K = np.empty(edges.shape[0])

    # add the flow to their orthogonal edge
    for i, (u, v) in enumerate(simplex_edges):
        K[i] = -flowdict[u][v] + flowdict[v][u]

    # derive correct gradients
    psi_diff += K * 2 * np.pi

    # construct temporary dict that hold all edge gradients from different direction
    psi_dict = {i: {} for i in range(num_points)}
    for diff, (u, v) in zip(psi_diff, edges):
        psi_dict[u][v] = diff
        psi_dict[v][u] = -diff

    # integrate the gradients
    result = psi.copy()
    for u, v in traverse_path:
        result[v] = result[u] + psi_dict[u][v]
    return result
