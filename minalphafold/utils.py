import torch

def dropout_rowwise(x, p, training):
    """Mask shared across dim 2 (columns)."""
    if not training or p == 0.0:
        return x
    mask_shape = (x.shape[0], x.shape[1], 1, x.shape[3])
    mask = x.new_ones(mask_shape)
    mask = torch.nn.functional.dropout(mask, p=p, training=True)
    return x * mask

def dropout_columnwise(x, p, training):
    """Mask shared across dim 1 (rows)."""
    if not training or p == 0.0:
        return x
    mask_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
    mask = x.new_ones(mask_shape)
    mask = torch.nn.functional.dropout(mask, p=p, training=True)
    return x * mask