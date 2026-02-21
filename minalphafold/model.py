import torch
from evoformer import Evoformer
from structure_module import StructureModule
from embedders import InputEmbedder, TemplatePair, TemplatePointwiseAttention, ExtraMsaStack
from utils import distance_bin
from random import randint

class AlphaFold2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.evoformer_blocks = torch.nn.ModuleList([Evoformer(config) for _ in range(config.num_evoformer)])
        self.structure_model = StructureModule(config)

        self.input_embedder = InputEmbedder(config)

        # Recycling embedders — single rep uses c_m (MSA channel dimension)
        self.recycle_norm_s = torch.nn.LayerNorm(config.c_m)
        self.recycle_norm_z = torch.nn.LayerNorm(config.c_z)
        self.recycle_linear_s = torch.nn.Linear(config.c_m, config.c_m)
        self.recycle_linear_z = torch.nn.Linear(config.c_z, config.c_z)
        self.recycle_linear_d = torch.nn.Linear(config.n_dist_bins, config.c_z)

        # Project from MSA channel dim (c_m) to single rep dim (c_s)
        self.single_rep_proj = torch.nn.Linear(config.c_m, config.c_s)

        # Template processing
        self.template_pair_feat_linear = torch.nn.Linear(88, config.c_t)
        self.template_pair_stack = TemplatePair(config)
        self.template_pointwise_att = TemplatePointwiseAttention(config)

        self.template_angle_linear_1 = torch.nn.Linear(51, config.c_m)
        self.template_angle_linear_2 = torch.nn.Linear(config.c_m, config.c_m)

        # Extra MSA processing
        self.extra_msa_feat_linear = torch.nn.Linear(25, config.c_e)
        self.extra_msa_blocks = torch.nn.ModuleList(
            [ExtraMsaStack(config) for _ in range(config.num_extra_msa)]
        )

        self.config = config

    def forward(
            self,
            target_feat: torch.Tensor,
            residue_index: torch.Tensor,
            msa_feat: torch.Tensor,
            extra_msa_feat: torch.Tensor,
            template_pair_feat: torch.Tensor,
            aatype: torch.Tensor,
            n_cycles: int = 3,
            n_ensemble: int = 1,
        ):
        assert(n_ensemble > 0)
        assert(n_cycles > 0)

        if self.training:
            n_cycles = randint(1, 4)

        outer_grad = torch.is_grad_enabled()

        N_res = target_feat.shape[1]
        c_m = self.config.c_m
        c_z = self.config.c_z
        batch_size = target_feat.shape[0]

        # Initialize recycling tensors (only once, before the loop)
        single_rep_prev = torch.zeros(batch_size, N_res, c_m, device=msa_feat.device)
        z_prev = torch.zeros(batch_size, N_res, N_res, c_z, device=msa_feat.device)
        x_prev = torch.zeros(batch_size, N_res, 3, device=msa_feat.device)

        for i in range(n_cycles):
            is_last = (i == n_cycles-1)

            with torch.set_grad_enabled(is_last and outer_grad):
                # Ensemble: accumulate non-MSA representations and average
                single_rep_accum = torch.zeros(batch_size, N_res, c_m, device=msa_feat.device)
                pair_repr_accum = torch.zeros(batch_size, N_res, N_res, c_z, device=msa_feat.device)

                msa_repr = None  # will be set by ensemble loop (n_ensemble > 0)
                for _ in range(n_ensemble):
                    msa_representation, pair_representation = self.input_embedder(target_feat, residue_index, msa_feat)

                    msa_repr = msa_representation.clone()
                    pair_repr = pair_representation.clone()

                    msa_repr[:, 0, :, :] += self.recycle_linear_s(self.recycle_norm_s(single_rep_prev))
                    pair_repr += self.recycle_linear_z(self.recycle_norm_z(z_prev))
                    pair_repr += self.recycle_linear_d(distance_bin(x_prev, self.config.n_dist_bins))

                    # Extra MSA processing
                    extra_msa_repr = self.extra_msa_feat_linear(extra_msa_feat)
                    for extra_block in self.extra_msa_blocks:
                        extra_msa_repr, pair_repr = extra_block(extra_msa_repr, pair_repr)

                    # Template processing
                    template_pair = self.template_pair_feat_linear(template_pair_feat)
                    template_pair = self.template_pair_stack(template_pair)
                    pair_repr = pair_repr + self.template_pointwise_att(template_pair, pair_repr)

                    for block in self.evoformer_blocks:
                        msa_repr, pair_repr = block(msa_repr, pair_repr)

                    single_rep_accum += msa_repr[:, 0, :, :]
                    pair_repr_accum += pair_repr

                # Average across ensemble members
                msa_first_row = single_rep_accum / n_ensemble
                pair_repr = pair_repr_accum / n_ensemble

                single_rep = self.single_rep_proj(msa_first_row)

                structure_predictions = self.structure_model(single_rep, pair_repr, aatype)

                if is_last:
                    return structure_predictions, pair_repr, msa_repr, single_rep

                # Recycle: store pre-projection single rep (c_m) for next cycle
                single_rep_prev = msa_first_row.detach()
                z_prev = pair_repr.detach()

                # Use pseudo-beta positions for recycling distance features
                is_gly = (aatype == 7)
                cb_idx = torch.where(is_gly, 1, 4)  # CA=1 for GLY, CB=4 otherwise
                atom_coords = structure_predictions["atom14_coords"]
                x_prev = torch.gather(
                    atom_coords, 2,
                    cb_idx[:, :, None, None].expand(-1, -1, 1, 3),
                ).squeeze(2).detach()

        raise ValueError("n_cycles must be >= 0")
