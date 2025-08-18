# Author: Qidong Yang & Jonathan Giezendanner

from torch_geometric.nn import knn

def search_k_neighbors(target_points, source_points, k):
    # base_points: (n_b, n_features) : where the points go to
    # cand_points: (n_c, n_features): where the points come from

    edge_index = knn(source_points, target_points, k)[[1, 0], :]

    # edge_index: 0: source, 1: target

    return edge_index
