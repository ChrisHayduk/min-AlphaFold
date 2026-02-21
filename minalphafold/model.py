import torch
from evoformer import Evoformer
from structure_module import StructureModule
from embedders import InputEmbedder
from utils import distance_bin
from random import randint

class AlphaFold2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.evoformer_blocks = torch.nn.ModuleList([Evoformer(config) for _ in range(config.num_evoformer)])
        self.structure_model = StructureModule(config)

        self.input_embedder = InputEmbedder(config)

        self.recycle_norm_s = torch.nn.LayerNorm(config.c_s)
        self.recycle_norm_z = torch.nn.LayerNorm(config.c_z)
        self.recycle_linear_s = torch.nn.Linear(config.c_s, config.c_s)
        self.recycle_linear_z = torch.nn.Linear(config.c_z, config.c_z)
        self.recycle_linear_d = torch.nn.Linear(config.n_dist_bins, config.c_z)

        self.single_rep_proj = torch.nn.Linear(config.c_s, config.c_s)

        self.template_angle_linear_1 = torch.nn.Linear(51, config.c_m)
        self.template_angle_linear_2 = torch.nn.Linear(config.c_m, config.c_m)

        self.template_pair_feat_linear = torch.nn.Linear(88, config.c_t)

        self.extra_msa_feat_linear = torch.nn.Linear(25, config.c_e)

        self.config = config

    def forward(
            self, 
            target_feat: torch.Tensor, 
            residue_index: torch.Tensor, 
            msa_feat: torch.Tensor,
            extra_msa_feat: torch.Tensor,
            template_angle_feat: torch.Tensor,
            template_pair_feat: torch.Tensor,
            n_cycles: int = 3,
            n_ensemble: int = 3,
        ):
        assert(n_ensemble > 0)
        assert(n_cycles > 0)

        if self.training:
            n_cycles = randint(1, 4)

        outer_grad = torch.is_grad_enabled()

        N_res = target_feat.shape[1]
        c_s = self.config.c_s
        c_z = self.config.c_z
        batch_size = target_feat.shape[0]

        single_rep_prev = torch.zeros(batch_size, N_res, c_s, device=msa_feat.device)   # previous single repr
        z_prev = torch.zeros((batch_size, N_res, N_res, c_z))  # previous pair repr
        x_prev = torch.zeros(batch_size, N_res, 3, device=msa_feat.device)    # previous Ca positions

        for i in range(n_cycles):
            is_last = (i == n_cycles-1)

            single_rep_prev = torch.zeros(batch_size, N_res, c_s, device=msa_feat.device)   # previous single repr
            z_prev = torch.zeros((batch_size, N_res, N_res, c_z))  # previous pair repr
            x_prev = torch.zeros(batch_size, N_res, 3, device=msa_feat.device)    # previous Ca positions

            with torch.set_grad_enabled(is_last and outer_grad):
                for n in range(n_ensemble):
                    msa_representation, pair_representation = self.input_embedder(target_feat, residue_index, msa_feat)

                    msa_repr = msa_representation.clone()
                    pair_repr = pair_representation.clone()

                    msa_repr[:, 0, :, :] += self.recycle_linear_s(self.recycle_norm_s(single_rep_prev))
                    pair_repr += self.recycle_linear_z(self.recycle_norm_z(z_prev))
                    pair_repr += self.recycle_linear_d(distance_bin(x_prev, self.config.n_dist_bins))

                    for block in self.evoformer_blocks:
                        msa_repr, pair_repr = block(msa_repr, pair_repr)

                    single_rep = msa_repr[:, 0, :, :]

                    single_rep = self.single_rep_proj(single_rep)

                structure_predictions = self.structure_model(single_rep, pair_repr)

                if is_last:
                    return structure_predictions, pair_repr, msa_repr, single_rep

                single_rep_prev = single_rep.detach()
                z_prev = pair_repr.detach()
                x_prev = structure_predictions["final_translations"].detach()

        raise ValueError("n_cycles must be >= 0")