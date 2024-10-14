import torch


def search_k_neighbors(base_points, cand_points, k):
    # base_points: (n_b, n_features)
    # cand_points: (n_c, n_features)

    dis = torch.sum((base_points.unsqueeze(1) - cand_points.unsqueeze(0)) ** 2, dim=-1)
    _, inds = torch.topk(dis, k, dim=1, largest=False)

    n_b = base_points.size(0)

    j_inds = inds.view((1, -1))
    i_inds = (torch.arange(n_b).view((-1, 1)) * torch.ones((n_b, k))).view((1, -1))

    edge_index = torch.cat([j_inds, i_inds], dim=0)

    return edge_index