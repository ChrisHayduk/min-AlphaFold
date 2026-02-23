import torch

def dropout_rowwise(x, p, training):
    """Mask shared across dim 2 (columns)."""
    if not training or p == 0.0:
        return x
    mask_shape = (x.shape[0], x.shape[1], 1, x.shape[3])
    mask = x.new_ones(mask_shape)
    mask = torch.nn.functional.dropout(mask, p=p, training=True)
    return x * mask

def distance_bin(positions, n_bins, d_min=2.0, d_max=22.0):
    dists = torch.cdist(positions, positions)
    # AF2-style uniform-width bins: width = (d_max - d_min) / n_bins
    step = (d_max - d_min) / n_bins
    bin_edges = d_min + step * torch.arange(1, n_bins, device=positions.device, dtype=dists.dtype)
    bin_idx = torch.bucketize(dists, bin_edges)
    return torch.nn.functional.one_hot(bin_idx, n_bins).to(dtype=positions.dtype)

def dropout_columnwise(x, p, training):
    """Mask shared across dim 1 (rows)."""
    if not training or p == 0.0:
        return x
    mask_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
    mask = x.new_ones(mask_shape)
    mask = torch.nn.functional.dropout(mask, p=p, training=True)
    return x * mask