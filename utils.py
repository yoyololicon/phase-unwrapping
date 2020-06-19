import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, csgraph
import maxflow

def wrap_func(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def scipy_maxflow(G: nx.DiGraph, s, t, capacity='capacity', **kwargs):
    cap = nx.get_edge_attributes(G, capacity)
    edges = G.edges
    cap = [cap[e] for e in edges]

    edges = np.array(edges)
    sparse_G = csr_matrix((cap, (edges[:, 0], edges[:, 1])),
                                 shape=(edges.max() + 1,) * 2)

    res = csgraph.maximum_flow(sparse_G, s, t)

    R = nx.algorithms.flow.utils.build_residual_network(G, 'capacity')
    R.graph['flow_value'] = res.flow_value
    residual = res.residual

    for i, j in R.edges:
        R[i][j]['flow'] = residual[i, j]

    return R

