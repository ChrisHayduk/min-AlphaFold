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

    def forward(self, single_representation: torch.Tensor, pair_representation: torch.Tensor, rotations: torch.Tensor, translation: torch.Tensor):
        # single_rep shape: (batch, N_res, c_s)
        # pair_rep shape: (batch, N_seq, N_res, c_z)
        # rotations shape: (batch, N_res, 3, 3)
        # translations shape: (batch, N_res, 3)
        pass