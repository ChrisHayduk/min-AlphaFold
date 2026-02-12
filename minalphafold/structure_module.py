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
            in_features=self.total_dim 
                + self.num_heads * self.n_value_points * 3 
                + self.num_heads * self.n_value_points      # norms
                + self.num_heads * config.c_z,
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

        batch_size = single_representation.shape[0]
        N_res = single_representation.shape[1]

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
        diff = global_frame_q[:, :, None, :, :, :] - global_frame_k[:, None, :, :, :, :]

        gamma = torch.nn.functional.softplus(self.head_weights)

        # Squared distances summed over points and xyz: (batch, i, j, h)
        point_scores = -0.5 * gamma[None, None, None, :] * self.w_c * torch.sum(diff ** 2, dim=(-1, -2))

        # Shape (batch, N_res, N_res, self.num_heads)
        rep_scores = torch.einsum('bihd, bjhd -> bijh', Q_rep, K_rep) / math.sqrt(self.head_dim) + B

        # Shape (batch, N_res, N_res, self.num_heads)
        scores = self.w_l * (rep_scores + point_scores)

        # Shape (batch, N_res, N_res, self.num_heads)
        attention = torch.nn.functional.softmax(scores, dim=-2)

        # Output shape: (batch, N_res self.num_heads, c_z)
        output_pair = torch.einsum('bijh, bijd -> bihd', attention, pair_representation)

        # Output shape: (batch, self.num_heads, N_res, self.head_dim)
        output_rep = torch.einsum('bijh, bjhd -> bhid', attention, V_rep)

        # Output shape: (batch, N_res, self.num_heads, self.n_value_points, 3)
        global_values = torch.einsum('biop, bihqp -> bihqo', rotations, V_frames) + translation[:, :, None, None, :]

        temp = torch.einsum('bijh, bjhqo->bhiqo', attention, global_values)

        # R^T
        R_inv = rotations.transpose(-1, -2)                          
        t_inv = -torch.einsum('bira, bia -> bir', R_inv, translation)
        
        output_values = torch.einsum('biop, bhiqp -> bhiqo', R_inv, temp) + t_inv[:, None, :, None, :]

        # output_rep: (batch, h, N_res, head_dim) → (batch, N_res, h * head_dim)
        output_rep = output_rep.permute(0, 2, 1, 3).reshape(batch_size, N_res, -1)

        # output_values: (batch, h, N_res, n_value_points, 3) → (batch, N_res, h, n_value_points, 3)
        output_values = output_values.permute(0, 2, 1, 3, 4)

        # (batch, N_res, h * n_value_points * 3)
        output_values = output_values.reshape(batch_size, N_res, -1)

        # (batch, N_res, h, n_value_points)
        output_norms = torch.sqrt(torch.sum(output_values ** 2, dim=-1) + 1e-8)  

        # (batch, N_res, h * n_value_points)
        output_norms = output_norms.reshape(batch_size, N_res, -1)

        # output_pair: (batch, N_res, h, c_z) → (batch, N_res, h * c_z)
        output_pair = output_pair.reshape(batch_size, N_res, -1)

        # Final concatenation and projection
        output = torch.cat([output_rep, output_values, output_norms, output_pair], dim=-1)
        output = self.linear_output(output)

        return output
    
class BackboneUpdate(torch.nn.Module):
    def __init__(self, config):
        self.linear = torch.nn.Linear(in_features=config.c_s, out_features=6)

    def forward(self, single_representation: torch.Tensor):
        # output shape; (batch, N_res, 6)
        vals = self.linear(single_representation)

        # Rotation quaternions
        b = vals[:, :, 0]
        c = vals[:, :, 1]
        d = vals[:, :, 2]

        a = torch.ones_like(b)
        norm = torch.sqrt(1 + b**2 + c**2 + d**2)

        a = a / norm
        b = b / norm
        c = c / norm
        d = d / norm

        # Construct pairwise multiplications for rotation matrix
        aa = a*a
        bb = b*b
        cc = c*c
        dd = d*d

        ab = a*b
        ac = a*c
        ad = a*d

        bc = b*c
        bd = b*d

        cd = c*d

        # Construct rotation matrix entries
        r11 = aa + bb - cc - dd
        r12 = 2*bc - 2*ad
        r13 = 2*bd + 2*ac

        r21 = 2*bc + 2*ad
        r22 = aa - bb + cc - dd
        r23 = 2*cd - 2*ab

        r31 = 2*bd - 2*ac
        r32 = 2*cd + 2*ab
        r33 = aa - bb - cc + dd

        # Output shape: (batch, N_res, 3, 3)
        R = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1).reshape((single_representation.shape[0], single_representation.shape[1], 3, 3))

        t = vals[:, :, 3:]

        return R, t