import torch    
import math
class StructureModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, single_representation: torch.Tensor, pair_representation: torch.Tensor):
        pass

class InvariantPointAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.ipa_num_heads
        self.head_dim = config.ipa_c
        self.total_dim = self.head_dim * self.num_heads
        self.n_query_points = config.ipa_n_query_points
        self.n_value_points = config.ipa_n_value_points

        self.linear_q_rep = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_k_rep = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)
        self.linear_v_rep = torch.nn.Linear(in_features=config.c_s, out_features=self.total_dim, bias=False)

        self.linear_q_frames = torch.nn.Linear(in_features=config.c_s, out_features=3*self.num_heads*self.n_query_points, bias=False)
        self.linear_k_frames = torch.nn.Linear(in_features=config.c_s, out_features=3*self.num_heads*self.n_query_points, bias=False)
        self.linear_v_frames = torch.nn.Linear(in_features=config.c_s, out_features=3*self.num_heads*self.n_value_points, bias=False)

        self.linear_bias = torch.nn.Linear(in_features=config.c_z, out_features=self.num_heads, bias=False)

        self.linear_output = torch.nn.Linear(
            in_features=self.total_dim + self.num_heads * self.n_value_points * 3 + self.num_heads * config.c_z,
            out_features=config.c_s
        )

        self.w_c = math.sqrt(2/(9 * self.n_query_points))
        self.w_l = math.sqrt(1/3)

        self.head_weights = torch.nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, single_representation: torch.Tensor, pair_representation: torch.Tensor, rotations: torch.Tensor, translation: torch.Tensor):
        # single_rep shape: (batch, N_res, c_s)
        # pair_rep shape: (batch, N_res, N_res, c_z)
        # rotations shape: (batch, N_res, 3, 3)
        # translations shape: (batch, N_res, 3)

        # Shapes (batch, N_res, self.total_dim)
        Q_rep = self.linear_q_rep(single_representation)
        K_rep = self.linear_k_rep(single_representation)
        V_rep = self.linear_v_rep(single_representation)

        Q_rep = Q_rep.reshape((Q_rep.shape[0], Q_rep.shape[1], self.num_heads, self.head_dim))
        K_rep = K_rep.reshape((K_rep.shape[0], K_rep.shape[1], self.num_heads, self.head_dim))
        V_rep = V_rep.reshape((V_rep.shape[0], V_rep.shape[1], self.num_heads, self.head_dim))

        # Shapes (batch, N_res, 3*self.num_heads*self.n_query_points)
        Q_frames = self.linear_q_frames(single_representation)
        K_frames = self.linear_k_frames(single_representation)

        Q_frames = Q_frames.reshape((Q_frames.shape[0], Q_frames.shape[1], self.num_heads, self.n_query_points, -1))
        K_frames = K_frames.reshape((K_frames.shape[0], K_frames.shape[1], self.num_heads, self.n_query_points, -1))

        # Shape (batch, N_res, 3*self.num_heads*self.n_value_points)
        V_frames = self.linear_v_frames(single_representation)
        V_frames = V_frames.reshape((V_frames.shape[0], V_frames.shape[1], self.num_heads, self.n_value_points, -1))

        # Shape (batch, N_res, N_res, self.num_heads)
        B = self.linear_bias(pair_representation)

        # Query frames shape: (batch, N_res, self.num_heads, self.n_query_points, 3)
        # Rotations shape: (batch, N_res, 3, 3)
        # Translation shape: (batch, N_res, 3)
        # Output shape: (batch, N_res, self.num_heads, self.n_query_points, 3)
        global_frame_q = torch.einsum('biop, bihqp -> bihqo', rotations, Q_frames) + translation[:, :, None, None, :]

        global_frame_k = torch.einsum('biop, bihqp -> bihqo', rotations, K_frames) + translation[:, :, None, None, :]

        # Difference: (batch, N_res_i, N_res_j, num_heads, n_query_points, 3)
        diff = global_q[:, :, None, :, :, :] - global_k[:, None, :, :, :, :]

        gamma = torch.nn.functional.softplus(self.head_weights)

        # Squared distances summed over points and xyz: (batch, i, j, h)
        point_scores = -0.5 * (gamma**(self.num_heads)) * self.w_c * torch.sum(diff ** 2, dim=(-1, -2))

        # Shape (batch, N_res, N_res, self.num_heads)
        rep_scores = 1/math.sqrt(self.head_dim) * torch.einsum('bihd, bjhd -> bijh', Q_rep, K_rep) / math.sqrt(self.head_dim) + B

        # Shape (batch, N_res, N_res, self.num_heads)
        scores = self.w_l * (rep_scores + point_scores)

        # Shape (batch, N_res, N_res, self.num_heads)
        attention = torch.nn.functional.softmax(scores, dim=-2)

