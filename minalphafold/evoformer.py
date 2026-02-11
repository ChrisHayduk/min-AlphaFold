import torch
import math
class Evoformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, msa_representation: torch.Tensor, pair_representation: torch.Tensor):
        pass

class MSARowAttentionWithPairBias(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_msa = torch.nn.LayerNorm(config.c_s)
        self.layer_norm_pair = torch.nn.LayerNorm(config.c_z)

        self.head_dim = config.dim
        self.num_heads = config.num_heads

        self.total_dim = self.head_dim * self.num_heads

        self.linear_q = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_k = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_v = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)

        self.linear_pair = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_gate = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim)

        self.linear_output = torch.nn.Linear(in_features=self.total_dim, out_features=config.c_s)

    def forward(self, msa_representation: torch.Tensor, pair_representation: torch.Tensor):
        msa_representation = self.layer_norm_msa(msa_representation)

        # Shape (batch, N_seq, N_res, self.total_dim)
        Q = self.linear_q(msa_representation)
        K = self.linear_k(msa_representation)
        V = self.linear_v(msa_representation)

        # Reshape to (batch, N_seq, N_res, self.num_heads, self.head_dim)
        Q = Q.reshape((Q.shape[0], Q.shape[1], Q.shape[2], self.num_heads, self.head_dim))
        K = K.reshape((K.shape[0], K.shape[1], K.shape[2], self.num_heads, self.head_dim))
        V = V.reshape((V.shape[0], V.shape[1], V.shape[2], self.num_heads, self.head_dim))

        G = self.linear_gate(msa_representation)
        G = G.reshape((G.shape[0], G.shape[1], G.shape[2], self.num_heads, self.head_dim))

        # Squash values in range 0 to 1 to act as gating mechanism
        G = torch.nn.functional.sigmoid(G)

        pair_representation = self.layer_norm_pair(pair_representation)

        # Shape (batch, N_res, N_res, self.num_heads)
        B = self.linear_pair(pair_representation)
        B = B.reshape((B.shape[0], B.shape[1], B.shape[2], self.num_heads, self.head_dim))

        # Shape (batch, 1, N_res, N_res, self.num_heads)
        B = B.unsqueeze(1)

        # Shape (batch, 1, self.num_heads, N_res, N_res)
        B = B.permute(0, 1, 4, 2, 3)

        # Shape (batch, N_seq, self.num_heads, N_res, N_res)
        scores = torch.einsum('bsihd, bsjhd -> bshij', Q, K)
        scores = scores / math.sqrt(self.head_dim) + B

        attention = torch.nn.functional.softmax(scores, dim=-1)

        values = torch.einsum('bshij, bsjhd -> bsihd', attention, V)

        values = G * values

        values = values.reshape((Q.shape[0], Q.shape[1], Q.shape[2], -1))

        output = self.linear_output(values)

        return output

class MSAColumnAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, msa_representation: torch.Tensor):
        pass

class MSATransition(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, msa_representation: torch.Tensor):
        pass

class OuterProductMean(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, msa_representation: torch.Tensor):
        pass

class TriangleMultiplicationOutgoing(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, pair_representation: torch.Tensor):
        pass

class TriangleMultiplicationIncoming(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, pair_representation: torch.Tensor):
        pass

class TriangleAttentionStartingNode(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, pair_representation: torch.Tensor):
        pass

class TriangleAttentionEndingNode(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, pair_representation: torch.Tensor):
        pass

class PairTransition(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, pair_representation: torch.Tensor):
        pass