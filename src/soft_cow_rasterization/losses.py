from __future__ import annotations

import torch

try:
    from pytorch3d.loss import (
        mesh_edge_loss,
        mesh_laplacian_smoothing,
        mesh_normal_consistency,
    )
except ImportError:
    mesh_edge_loss = None
    mesh_laplacian_smoothing = None
    mesh_normal_consistency = None


def require_pytorch3d_losses():
    if (
        mesh_edge_loss is None
        or mesh_laplacian_smoothing is None
        or mesh_normal_consistency is None
    ):
        raise RuntimeError(
            "当前环境缺少 pytorch3d.loss，无法计算网格正则项。"
        )


def silhouette_loss(rendered_silhouette: torch.Tensor, target_silhouette: torch.Tensor) -> torch.Tensor:
    return torch.mean((rendered_silhouette - target_silhouette) ** 2)


def regularization_loss(
    mesh,
    w_lap: float = 0.10,
    w_edge: float = 0.80,
    w_normal: float = 0.01,
):
    require_pytorch3d_losses()

    loss_lap = mesh_laplacian_smoothing(mesh, method="uniform")
    loss_edge = mesh_edge_loss(mesh)
    loss_normal = mesh_normal_consistency(mesh)

    loss_reg = w_lap * loss_lap + w_edge * loss_edge + w_normal * loss_normal
    return loss_reg, loss_lap, loss_edge, loss_normal
