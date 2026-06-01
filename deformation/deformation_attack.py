"""
This script implements the deformation-based targeted adversarial attack.
"""

import torch
import torch.nn as nn

from tqdm import tqdm
from utils.utils import make_coordinate_tensor
from deformation.cubspline_3d import bspline_interp3d


# ======================================================================================================================
# Regularizers
# ======================================================================================================================
def smoothness_reg(u: torch.Tensor) -> torch.Tensor:
    """
    Smoothness regularizer that penalizes first-order spatial gradients.
    :param u: displacement field with shape [B, X, Y, Z, D]
    :return: smoothness loss
    """
    dx = u[:, 1:, :, :, :] - u[:, :-1, :, :, :]
    dy = u[:, :, 1:, :, :] - u[:, :, :-1, :, :]
    dz = u[:, :, :, 1:, :] - u[:, :, :, :-1, :]

    return dx.pow(2).mean() + dy.pow(2).mean() + dz.pow(2).mean()

def bending_reg(u: torch.Tensor) -> torch.Tensor:
    """
    Bending energy regularizer that penalizes second-order spatial gradients.
    :param u: displacement field with shape [B, X, Y, Z, D]
    :return: bending energy loss
    """
    # Pure second derivatives
    ddx = u[:, 2:, :, :, :] - 2 * u[:, 1:-1, :, :, :] + u[:, :-2, :, :, :]
    ddy = u[:, :, 2:, :, :] - 2 * u[:, :, 1:-1, :, :] + u[:, :, :-2, :, :]
    ddz = u[:, :, :, 2:, :] - 2 * u[:, :, :, 1:-1, :] + u[:, :, :, :-2, :]

    # Mixed second derivatives
    dxy = u[:, 1:, 1:, :, :] - u[:, 1:, :-1, :, :] - u[:, :-1, 1:, :, :] + u[:, :-1, :-1, :, :]
    dxz = u[:, 1:, :, 1:, :] - u[:, 1:, :, :-1, :] - u[:, :-1, :, 1:, :] + u[:, :-1, :, :-1, :]
    dyz = u[:, :, 1:, 1:, :] - u[:, :, 1:, :-1, :] - u[:, :, :-1, 1:, :] + u[:, :, :-1, :-1, :]

    return (ddx.pow(2).mean() + ddy.pow(2).mean() + ddz.pow(2).mean() +
            2 * (dxy.pow(2).mean() + dxz.pow(2).mean() + dyz.pow(2).mean()))

# ======================================================================================================================
# Loss function
# ======================================================================================================================
def swm_loss(
        logits_list: list[torch.Tensor],
        target_class: int,
        margin: float = 4.5,
        temperature: float = 1.0
) -> torch.Tensor:
    """
    Smooth worst-case margin loss using log-sum-exp (soft min/max).

    :param logits_list: a list of logits
    :param target_class: the target class label
    :param margin: logit margin
    :param temperature: temperature parameter (for temperature -> 0, approaches hard min/max)
    :return: swm loss
    """
    stacked = torch.stack([l.squeeze() for l in logits_list])

    if target_class == 1:
        # Want logits > margin, so find soft-min (worst case)
        # soft-min(x) = -logsumexp(-x / temp) * temp
        soft_worst = -temperature * torch.logsumexp(-stacked / temperature, dim=0)
        return torch.relu(margin - soft_worst)
    else:
        # Want logits < -margin, so find soft-max (worst case)
        # soft-max(x) = logsumexp(x / temp) * temp
        soft_worst = temperature * torch.logsumexp(stacked / temperature, dim=0)
        return torch.relu(margin + soft_worst)

# ======================================================================================================================
# Diagnostics
# ======================================================================================================================
@torch.no_grad()
def jacobian_det(u: torch.Tensor) -> torch.Tensor:
    """
    Per-voxel Jacobian determinant of the deformation φ(x) = x + u(x).

    The displacement field is stored in normalized [-1, 1] coordinates over a
    regular voxel grid of size (X, Y, Z); a voxel index step of 1 corresponds
    to a normalized-coord step of 2/N. The scale factor N/2 converts the
    per-voxel finite-difference gradient into the gradient w.r.t. normalized
    coordinates, so that an undeformed field gives det(J) = 1.

    :param u: displacement field, shape [B, X, Y, Z, 3]
    :return: per-voxel determinant tensor, shape [B, X, Y, Z]
    """
    _, X, Y, Z, _ = u.shape
    scales = (X / 2.0, Y / 2.0, Z / 2.0)

    rows = []
    for c in range(3):
        gx, gy, gz = torch.gradient(u[..., c], dim=(1, 2, 3))
        rows.append(torch.stack(
            [gx * scales[0], gy * scales[1], gz * scales[2]], dim=-1
        ))
    nabla_u = torch.stack(rows, dim=-2)  # [B, X, Y, Z, 3, 3]
    eye = torch.eye(3, device=u.device).expand_as(nabla_u)
    return torch.linalg.det(eye + nabla_u)


# ======================================================================================================================
# Deformer
# ======================================================================================================================
class Deformer:
    def __init__(self,
                 img_size: tuple[int, int, int],
                 cp_spacing: int,
                 model_list: list[nn.Module],
                 lr: float,
                 device: torch.device,
                 margin: float = 4.5
                 ):
        """
        Deformer implements the base class used to define and optimize a deformation field.

        :param img_size: size of the input image
        :param cp_spacing: control point spacing (defines spacing between control points)
        :param model_list: list of models used to optimize the deformation field
        :param lr: learning rate
        :param device: device on which the deformation field should be optimized
        :param margin: target logit margin
        """

        self.device = device
        self.cp_spacing = cp_spacing
        self.lr = lr
        self.margin = margin

        """ Define the control point grid """
        grid_shape = [s // cp_spacing for s in img_size]
        self.cps = make_coordinate_tensor(grid_shape).to(device)
        self.dense_grid = make_coordinate_tensor(img_size).flip(-1).to(device)

        """ Define trainable offset for each control point (init as zero) """
        self.cpo = nn.Parameter(torch.zeros_like(self.cps, dtype=torch.float32, device=device))

        """ Define a mask that restricts which control point offsets can be optimized (optional) """
        self.mask = torch.ones_like(self.cpo, device=device)
        self.mask[:, :(self.mask.shape[1] // 2), :, :] = 0  # Mask out the posterior half of the skull

        """ Define the used optimizer (Adam) """
        self.optimizer = torch.optim.Adam(params=[self.cpo], lr=lr)
        self.scheduler = None  # Will be set in deform, once the total steps are known

        """ Define the classification network ensemble (all models are frozen) """
        self.models = model_list
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        self.model_weights = torch.ones(len(self.models), device=device) / len(self.models)  # Uniformly weight models

    @torch.no_grad()
    def jacobian_stats(self) -> dict:
        """
        Compute Jacobian-determinant statistics for the current control-point
        offsets. Useful as a folding / volume-preservation sanity check on
        the optimized deformation field.

        :return: dict with min, max, mean, std, neg_fraction (fraction of
                 voxels with det <= 0, i.e. folded), and the worst (most
                 negative or smallest) value.
        """
        cpo = torch.einsum('xyzd->dxyz', (self.cpo * self.mask))[None, ...]
        u = bspline_interp3d(cpo, scale_factor=self.cp_spacing)
        u = torch.einsum('bdxyz->bxyzd', u).flip(-1)
        det = jacobian_det(u)
        return {
            'det_J_min': det.min().item(),
            'det_J_max': det.max().item(),
            'det_J_mean': det.mean().item(),
            'det_J_std': det.std().item(),
            'det_J_neg_fraction': (det <= 0).float().mean().item(),
        }

    def deform(self,
               img: torch.Tensor,
               target_class: int,
               steps: int,
               log_step: int,
               record_trajectory: bool = False,
               ) -> torch.Tensor:
        """
        Function that performs the adversarial optimization for a single image.
        :param img: input image (to be deformed)
        :param target_class: target class (0: male, 1: female)
        :param steps: number of optimization steps
        :param log_step: how often to refresh the tqdm postfix stats
        :param record_trajectory: if True, record per-step loss / worst-logit /
                                  mean-prob into self.last_trajectory
        :return: deformed image
        """
        img = img.squeeze().to(self.device)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=steps, eta_min=1e-5)

        self.last_trajectory = (
            {'step': [], 'worst': [], 'mean_prob': [], 'swm': [], 'reg': []}
            if record_trajectory else None
        )

        pbar = tqdm(range(steps), desc="Optimizing", leave=False)
        for step in pbar:
            self.optimizer.zero_grad()

            """ Apply current deformation """
            # Displacement Field u
            cpo = torch.einsum('xyzd->dxyz', (self.cpo * self.mask))[None, ...]
            u = bspline_interp3d(cpo, scale_factor=self.cp_spacing)
            u = torch.einsum('bdxyz->bxyzd', u).flip(-1)
            u_copy = u.clone().detach().requires_grad_(True)

            # Obtain deformation field & apply to image
            def_field = self.dense_grid + u_copy
            def_img = torch.nn.functional.grid_sample(img[None, None], def_field, align_corners=True)
            def_img_copy = def_img.clone().detach().requires_grad_(True)

            """ Apply classification ensemble - collect all logits """
            ensemble_logits = []

            # Forward passes (& backward passes till deformed image copy)
            for model in self.models:
                logits = model(def_img_copy[:, :, :, 128:, :])
                ensemble_logits.append(logits)

                def_img_flipped = torch.flip(def_img_copy, dims=[2])
                logits_flipped = model(def_img_flipped[:, :, :, 128:, :])
                ensemble_logits.append(logits_flipped)

            """ Compute swm loss """
            L_swm = swm_loss(ensemble_logits, target_class, margin=self.margin, temperature=1.0)
            L_swm.backward()
            def_img.backward(def_img_copy.grad)

            """ Regularization """
            L_smooth = smoothness_reg(u_copy) * 1e8
            L_bend = bending_reg(u_copy) * 1e8
            L_reg = L_smooth + L_bend
            L_reg.backward()

            # Backward pass to control point offsets
            u.backward(u_copy.grad)

            self.optimizer.step()
            self.scheduler.step()

            """ Live status in the progress bar (no scrollback spam) """
            need_postfix = (step % log_step == 0 or step == steps - 1)
            if need_postfix or record_trajectory:
                with torch.no_grad():
                    logits_vals = [l.item() for l in ensemble_logits]
                    mean_prob = sum(torch.sigmoid(l).item() for l in ensemble_logits) / len(ensemble_logits)
                worst = min(logits_vals) if target_class == 1 else max(logits_vals)

            if need_postfix:
                pbar.set_postfix({
                    'swm': f"{L_swm.item():.3f}",
                    'reg': f"{L_reg.item():.3f}",
                    'worst': f"{worst:+.2f}",
                    'μ_p': f"{mean_prob:.3f}",
                })

            if record_trajectory:
                self.last_trajectory['step'].append(step)
                self.last_trajectory['worst'].append(worst)
                self.last_trajectory['mean_prob'].append(mean_prob)
                self.last_trajectory['swm'].append(L_swm.item())
                self.last_trajectory['reg'].append(L_reg.item())

            """ Periodically clear cache """
            if step % 5 == 0:
                torch.cuda.empty_cache()

        return def_img