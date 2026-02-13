import torch

class TorsionAngleLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, torsion_angles: torch.Tensor, torsion_angles_true: torch.Tensor, torsion_angles_true_alt: torch.Tensor):
        # torsion_angles shape: (batch, N_res, 7, 2)

        norm = torch.sqrt(torch.sum(torsion_angles**2, dim=-1, keepdim=True))

        torsion_angles = torsion_angles / norm

        true_dist = torch.sqrt(torch.sum((torsion_angles_true - torsion_angles)**2, dim=-1, keepdim=True))

        alt_true_dist = torch.sqrt(torch.sum((torsion_angles_true_alt - torsion_angles)**2, dim=-1, keepdim=True))

        torsion_loss = torch.mean(torch.minimum(true_dist, alt_true_dist), dim=(2, 3))

        angle_norm_loss = torch.mean(torch.abs(norm - 1), dim=(2, 3))

        return torsion_loss + 0.02 * angle_norm_loss
    
class FAPELoss(torch.nn.Module):
    def __init__(self, d_clamp=10.0, eps=1e-4, Z=10.0):
        super().__init__()
        self.eps = eps
        self.d_clamp_val = d_clamp
        self.Z = Z

    def forward(self, 
                predicted_rotations,      # (b, N_res, 3, 3)
                predicted_translations,   # (b, N_res, 3)
                predicted_atom_positions, # (b, N_atoms, 3)
                true_rotations,           # (b, N_res, 3, 3)
                true_translations,        # (b, N_res, 3)
                true_atom_positions       # (b, N_atoms, 3)
    ):
        # Predicted inverse frames
        R_pred_inv = predicted_rotations.transpose(-1, -2)
        t_pred_inv = -torch.einsum('birc, bic -> bir', R_pred_inv, predicted_translations)

        # True inverse frames
        R_true_inv = true_rotations.transpose(-1, -2)
        t_true_inv = -torch.einsum('birc, bic -> bir', R_true_inv, true_translations)

        # Project ALL atoms through ALL frames (cross-product)
        # Result: (b, N_frames, N_atoms, 3)
        x_frames_pred = torch.einsum('biop, bjp -> bijo', R_pred_inv, predicted_atom_positions) \
                         + t_pred_inv[:, :, None, :]
        x_frames_true = torch.einsum('biop, bjp -> bijo', R_true_inv, true_atom_positions) \
                         + t_true_inv[:, :, None, :]

        # Distance: (b, N_frames, N_atoms)
        dist = torch.sqrt(
            torch.sum((x_frames_pred - x_frames_true) ** 2, dim=-1) + self.eps
        )

        # Clamp and average
        dist_clamped = torch.clamp(dist, max=self.d_clamp_val)

        fape_loss = (1.0 / self.Z) * torch.mean(dist_clamped, dim=(-1, -2))

        return fape_loss
    
class PLDDTLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_plddt: torch.Tensor, true_plddt: torch.Tensor):
        # Input shapes: (batch, N_res, n_plddt_bins)
        
        log_pred = torch.log_softmax(pred_plddt, dim=-1)

        # Per-residue cross-entropy: (batch, N_res)
        conf_loss = -torch.einsum('bic, bic -> bi', true_plddt, log_pred)

        # Mean over residues: (batch,)
        conf_loss = torch.mean(conf_loss, dim=-1)

        return conf_loss
    
class DistogramLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_distograms: torch.Tensor, true_distograms: torch.Tensor):
        # input shapes: (batch, N_res, N_res, num_dist_buckets)

        log_pred = torch.log_softmax(pred_distograms, dim=-1)

        vals = torch.einsum('bijc, bijc -> bij', true_distograms, log_pred)

        dist_loss = - torch.mean(vals, dim=(1,2))

        return dist_loss

class MSALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, msa_preds: torch.Tensor, msa_true: torch.Tensor, msa_mask: torch.Tensor):
        # msa_preds: (batch, N_seq, N_res, n_msa_classes) — raw logits
        # msa_true:  (batch, N_seq, N_res, n_msa_classes) — one-hot targets
        # msa_mask:  (N_seq, N_res) — 1 for masked positions to predict, 0 otherwise

        log_pred = torch.log_softmax(msa_preds, dim=-1)

        # Per-position cross-entropy: (batch, N_seq, N_res)
        ce = -torch.einsum('bsic, bsic -> bsi', msa_true, log_pred)

        # Apply mask: (N_seq, N_res) -> (1, N_seq, N_res)
        mask = msa_mask.unsqueeze(0)
        ce = ce * mask

        # Average over masked positions only
        N_mask = torch.sum(mask).clamp(min=1)
        msa_loss = torch.sum(ce, dim=(1, 2)) / N_mask

        return msa_loss
    
class ExperimentallyResolvedLoss(torch.nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, ground_truth: torch.Tensor):
        # preds shape: (batch, N_res, 14)
        # groud_truth shape: (batch, N_res, 14) - binary, 1 if atom is exp resolved, 0 if not

        probs = torch.sigmoid(preds)

        log_probs = torch.log(probs + self.eps)
        log_inv_probs = torch.log(1 - probs + self.eps)

        exp_resolved_loss = torch.mean(- ground_truth * log_probs - (1 - ground_truth) * log_inv_probs, dim=(1,2))

        return exp_resolved_loss