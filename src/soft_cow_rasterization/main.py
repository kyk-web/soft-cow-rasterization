from __future__ import annotations

import csv
from collections import defaultdict
import os
import warnings
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .losses import regularization_loss, silhouette_loss
from .shader import build_renderer

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_MESH_PATH = ASSETS_DIR / "cow.obj"
TARGET_IMAGE_PATH = DATA_DIR / "target.png"

IMAGE_SIZE = int(os.getenv("SCR_IMAGE_SIZE", "96"))
NUM_EPOCHS = int(os.getenv("SCR_EPOCHS", "300"))
LEARNING_RATE = float(os.getenv("SCR_LR", "0.20"))
ICO_LEVEL = int(os.getenv("SCR_ICO_LEVEL", "3"))

W_LAP = 0.02
W_EDGE = 0.20
W_NORMAL = 0.01

VIEW_ELEV = [0.0]
VIEW_AZIM = [180.0]
SNAPSHOT_EPOCHS = {0, 25, 50, 75, 100, 150, 200, 250, 299}
FALLBACK_IMAGE_SIZE = 128
FALLBACK_RINGS = 9
FALLBACK_SECTORS = 40
FALLBACK_SIGMA = 0.035
FALLBACK_LR = 0.03


def require_pytorch3d():
    try:
        from pytorch3d.io import load_objs_as_meshes, save_obj
        from pytorch3d.structures import Meshes
        from pytorch3d.utils import ico_sphere
        from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
    except ImportError as exc:
        raise RuntimeError(
            "当前环境缺少 pytorch3d，无法按作业要求执行 SoftSilhouetteShader 网格优化。"
            "请先安装 pytorch3d，再运行 `python -m soft_cow_rasterization.main`。"
        ) from exc

    return {
        "FoVPerspectiveCameras": FoVPerspectiveCameras,
        "Meshes": Meshes,
        "ico_sphere": ico_sphere,
        "load_objs_as_meshes": load_objs_as_meshes,
        "look_at_view_transform": look_at_view_transform,
        "save_obj": save_obj,
    }


def has_pytorch3d() -> bool:
    try:
        require_pytorch3d()
    except RuntimeError:
        return False
    return True


def load_target_image(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"找不到目标剪影图: {path}")

    img = Image.open(path).convert("L")
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def normalize_verts(verts: torch.Tensor) -> torch.Tensor:
    centered = verts - verts.mean(dim=0, keepdim=True)
    scale = centered.abs().amax().clamp(min=1e-6)
    return centered / scale


def normalize_mesh(mesh, Meshes):
    verts = normalize_verts(mesh.verts_packed())
    faces = mesh.faces_packed()
    return Meshes(verts=[verts], faces=[faces])


def build_cameras(device: torch.device, elevation: list[float], azimuth: list[float]):
    p3d = require_pytorch3d()
    R, T = p3d["look_at_view_transform"](
        dist=2.7,
        elev=elevation,
        azim=azimuth,
        device=device,
    )
    return p3d["FoVPerspectiveCameras"](device=device, R=R, T=T)


def render_silhouettes(renderer, mesh, cameras) -> torch.Tensor:
    batch_size = cameras.R.shape[0]
    images = renderer(mesh.extend(batch_size), cameras=cameras)
    return images[..., 3]


def load_target_silhouettes(renderer, device: torch.device):
    p3d = require_pytorch3d()

    if TARGET_MESH_PATH.exists():
        target_mesh = p3d["load_objs_as_meshes"]([str(TARGET_MESH_PATH)], device=device)
        target_mesh = normalize_mesh(target_mesh, p3d["Meshes"])
        cameras = build_cameras(device, VIEW_ELEV, VIEW_AZIM)
        target_silhouettes = render_silhouettes(renderer, target_mesh, cameras)
        reference_view = target_silhouettes[0]
        source_note = f"target mesh: {TARGET_MESH_PATH}"
        return target_silhouettes, cameras, reference_view, source_note

    target_image = load_target_image(TARGET_IMAGE_PATH, IMAGE_SIZE, device)
    cameras = build_cameras(device, [VIEW_ELEV[0]], [VIEW_AZIM[0]])
    source_note = f"target silhouette: {TARGET_IMAGE_PATH}"
    warnings.warn(
        "未找到 assets/cow.obj，当前使用 data/target.png 作为目标轮廓。"
        "优化流程仍然基于 PyTorch3D 的 SoftSilhouetteShader，"
        "但目标监督为单视角剪影而非多视角牛网格。",
        stacklevel=2,
    )
    return target_image.unsqueeze(0), cameras, target_image, source_note


def build_source_mesh(device: torch.device):
    p3d = require_pytorch3d()
    source_mesh = p3d["ico_sphere"](ICO_LEVEL, device)
    return normalize_mesh(source_mesh, p3d["Meshes"])


def save_snapshot(
    target_view: torch.Tensor,
    pred_view: torch.Tensor,
    epoch: int,
    total_loss: float,
    silhouette_err: float,
    save_path: Path | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8))

    axes[0].imshow(target_view.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Ground Truth Silhouette")
    axes[0].axis("off")

    axes[1].imshow(pred_view.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Optimizing... (Epoch {epoch})")
    axes[1].axis("off")

    fig.suptitle(
        f"迭代步数: {epoch}/{NUM_EPOCHS} | 总 Loss: {total_loss:.4f} | 剪影误差: {silhouette_err:.4f}",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout()

    save_path = save_path or OUTPUT_DIR / f"epoch_{epoch:03d}.png"
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_loss_curve(history: list[dict[str, float]]) -> Path:
    steps = [item["epoch"] for item in history]
    total = [item["total"] for item in history]
    sil = [item["silhouette"] for item in history]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(steps, total, label="total loss", linewidth=2.0)
    ax.plot(steps, sil, label="silhouette loss", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Optimization Loss Curve")
    ax.grid(alpha=0.25)
    ax.legend()

    save_path = OUTPUT_DIR / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def save_loss_csv(history: list[dict[str, float]]) -> Path:
    save_path = OUTPUT_DIR / "loss_log.csv"
    with save_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["epoch", "total", "silhouette", "laplacian", "edge", "normal"],
        )
        writer.writeheader()
        writer.writerows(history)
    return save_path


def save_final_mesh(mesh) -> Path:
    p3d = require_pytorch3d()
    save_path = OUTPUT_DIR / "final_deformed_mesh.obj"
    p3d["save_obj"](
        f=save_path,
        verts=mesh.verts_packed().detach().cpu(),
        faces=mesh.faces_packed().detach().cpu(),
    )
    return save_path


def create_fallback_source_mesh(device: torch.device):
    verts = [[0.0, 0.0, 0.38]]
    for ring in range(1, FALLBACK_RINGS + 1):
        radius = 0.62 * ring / FALLBACK_RINGS
        z = 0.38 * np.sqrt(max(0.0, 1.0 - (ring / FALLBACK_RINGS) ** 2))
        for sector in range(FALLBACK_SECTORS):
            theta = 2.0 * np.pi * sector / FALLBACK_SECTORS
            verts.append([radius * np.cos(theta), radius * np.sin(theta), z])

    faces = []
    for sector in range(FALLBACK_SECTORS):
        faces.append([0, 1 + sector, 1 + ((sector + 1) % FALLBACK_SECTORS)])

    for ring in range(2, FALLBACK_RINGS + 1):
        prev_start = 1 + (ring - 2) * FALLBACK_SECTORS
        curr_start = 1 + (ring - 1) * FALLBACK_SECTORS
        for sector in range(FALLBACK_SECTORS):
            p0 = prev_start + sector
            p1 = prev_start + ((sector + 1) % FALLBACK_SECTORS)
            c0 = curr_start + sector
            c1 = curr_start + ((sector + 1) % FALLBACK_SECTORS)
            faces.append([p0, c0, c1])
            faces.append([p0, c1, p1])

    verts_t = torch.tensor(verts, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces, dtype=torch.long, device=device)
    return verts_t, faces_t


def build_edges_and_adjacent_faces(faces: torch.Tensor):
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, face in enumerate(faces.detach().cpu().tolist()):
        tri_edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ]
        for edge in tri_edges:
            edge_to_faces[edge].append(face_id)

    edges = torch.tensor(list(edge_to_faces.keys()), dtype=torch.long, device=faces.device)
    adjacent_pairs = [
        pair for pair in edge_to_faces.values() if len(pair) == 2
    ]
    adjacent_pairs_t = torch.tensor(adjacent_pairs, dtype=torch.long, device=faces.device)
    return edges, adjacent_pairs_t


def make_pixel_grid(image_size: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, device=device),
        torch.linspace(-1.0, 1.0, image_size, device=device),
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def point_segment_distance(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    ap = points.unsqueeze(0) - a.unsqueeze(1)
    ab = (b - a).unsqueeze(1)
    t = (ap * ab).sum(dim=-1) / ((ab * ab).sum(dim=-1) + eps)
    t = t.clamp(0.0, 1.0)
    closest = a.unsqueeze(1) + t.unsqueeze(-1) * ab
    return torch.sqrt(((points.unsqueeze(0) - closest) ** 2).sum(dim=-1) + eps)


def soft_rasterize_silhouette(
    verts: torch.Tensor,
    faces: torch.Tensor,
    grid: torch.Tensor,
    image_size: int,
    sigma: float,
    chunk_size: int = 96,
) -> torch.Tensor:
    xy = verts[:, :2]
    transmittance = torch.ones(grid.shape[0], device=verts.device)

    for start in range(0, faces.shape[0], chunk_size):
        face_chunk = faces[start : start + chunk_size]
        tri = xy[face_chunk]
        v0 = tri[:, 0]
        v1 = tri[:, 1]
        v2 = tri[:, 2]

        orient = cross2(v1 - v0, v2 - v0)
        e0 = cross2(v1.unsqueeze(1) - v0.unsqueeze(1), grid.unsqueeze(0) - v0.unsqueeze(1))
        e1 = cross2(v2.unsqueeze(1) - v1.unsqueeze(1), grid.unsqueeze(0) - v1.unsqueeze(1))
        e2 = cross2(v0.unsqueeze(1) - v2.unsqueeze(1), grid.unsqueeze(0) - v2.unsqueeze(1))

        inside = torch.where(
            orient.unsqueeze(1) >= 0,
            (e0 >= 0) & (e1 >= 0) & (e2 >= 0),
            (e0 <= 0) & (e1 <= 0) & (e2 <= 0),
        )

        d01 = point_segment_distance(grid, v0, v1)
        d12 = point_segment_distance(grid, v1, v2)
        d20 = point_segment_distance(grid, v2, v0)
        distance = torch.minimum(torch.minimum(d01, d12), d20)
        signed_distance = torch.where(inside, -distance, distance)
        alpha = torch.sigmoid(-signed_distance / sigma).clamp(0.0, 0.999)
        transmittance = transmittance * torch.prod(1.0 - alpha, dim=0)

    silhouette = 1.0 - transmittance
    return silhouette.view(image_size, image_size)


def laplacian_smoothing_loss(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    num_verts = verts.shape[0]
    neighbor_sum = torch.zeros_like(verts)
    degree = torch.zeros(num_verts, 1, device=verts.device)

    i = edges[:, 0]
    j = edges[:, 1]
    neighbor_sum.index_add_(0, i, verts[j])
    neighbor_sum.index_add_(0, j, verts[i])
    degree.index_add_(0, i, torch.ones_like(degree[i]))
    degree.index_add_(0, j, torch.ones_like(degree[j]))

    neighbor_mean = neighbor_sum / degree.clamp(min=1.0)
    return torch.mean((verts - neighbor_mean) ** 2)


def edge_length_consistency_loss(
    verts: torch.Tensor,
    edges: torch.Tensor,
    original_lengths: torch.Tensor,
) -> torch.Tensor:
    current_lengths = torch.norm(verts[edges[:, 0]] - verts[edges[:, 1]], dim=-1)
    return torch.mean(
        ((current_lengths - original_lengths) / original_lengths.clamp(min=1e-6)) ** 2
    )


def normal_consistency_loss(verts: torch.Tensor, faces: torch.Tensor, adjacent_pairs: torch.Tensor):
    if adjacent_pairs.numel() == 0:
        return verts.sum() * 0.0

    tri = verts[faces]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)
    n1 = normals[adjacent_pairs[:, 0]]
    n2 = normals[adjacent_pairs[:, 1]]
    return torch.mean(1.0 - (n1 * n2).sum(dim=-1))


def save_plain_obj(verts: torch.Tensor, faces: torch.Tensor) -> Path:
    save_path = OUTPUT_DIR / "final_deformed_mesh.obj"
    with save_path.open("w", encoding="utf-8") as fp:
        for vert in verts.detach().cpu().numpy():
            fp.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in faces.detach().cpu().numpy():
            fp.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return save_path


def optimize_fallback():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_view = load_target_image(TARGET_IMAGE_PATH, FALLBACK_IMAGE_SIZE, device)
    grid = make_pixel_grid(FALLBACK_IMAGE_SIZE, device)

    base_verts, faces = create_fallback_source_mesh(device)
    edges, adjacent_pairs = build_edges_and_adjacent_faces(faces)
    original_lengths = torch.norm(base_verts[edges[:, 0]] - base_verts[edges[:, 1]], dim=-1).detach()

    deform_verts = torch.zeros_like(base_verts, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([deform_verts], lr=FALLBACK_LR)

    history: list[dict[str, float]] = []
    gif_frames = []

    print(f"Using device: {device}")
    print(f"Target source: target silhouette: {TARGET_IMAGE_PATH}")
    print("PyTorch3D 不可用，当前使用兼容版软栅格化执行作业流程。")
    print(f"Output dir: {OUTPUT_DIR}")

    for epoch in tqdm(range(NUM_EPOCHS), desc="Optimizing"):
        optimizer.zero_grad()

        current_verts = base_verts + deform_verts
        pred_view = soft_rasterize_silhouette(
            verts=current_verts,
            faces=faces,
            grid=grid,
            image_size=FALLBACK_IMAGE_SIZE,
            sigma=FALLBACK_SIGMA,
        )

        loss_sil = torch.mean((pred_view - target_view) ** 2)
        loss_lap = laplacian_smoothing_loss(current_verts, edges)
        loss_edge = edge_length_consistency_loss(current_verts, edges, original_lengths)
        loss_normal = normal_consistency_loss(current_verts, faces, adjacent_pairs)
        loss = loss_sil + 0.002 * loss_lap + 0.004 * loss_edge + 0.01 * loss_normal

        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch,
                "total": float(loss.item()),
                "silhouette": float(loss_sil.item()),
                "laplacian": float(loss_lap.item()),
                "edge": float(loss_edge.item()),
                "normal": float(loss_normal.item()),
            }
        )

        if epoch in SNAPSHOT_EPOCHS:
            snapshot = save_snapshot(
                target_view=target_view,
                pred_view=pred_view,
                epoch=epoch,
                total_loss=float(loss.item()),
                silhouette_err=float(loss_sil.item()),
            )
            gif_frames.append(imageio.imread(snapshot))

            print(
                f"Epoch {epoch:03d} | Total: {loss.item():.4f} | "
                f"Silhouette: {loss_sil.item():.4f} | Lap: {loss_lap.item():.4f} | "
                f"Edge: {loss_edge.item():.4f} | Normal: {loss_normal.item():.4f}"
            )

    final_verts = base_verts + deform_verts
    final_pred = soft_rasterize_silhouette(
        verts=final_verts,
        faces=faces,
        grid=grid,
        image_size=FALLBACK_IMAGE_SIZE,
        sigma=FALLBACK_SIGMA,
    )

    comparison_path = OUTPUT_DIR / "final_comparison.png"
    save_snapshot(
        target_view=target_view,
        pred_view=final_pred,
        epoch=NUM_EPOCHS - 1,
        total_loss=history[-1]["total"],
        silhouette_err=history[-1]["silhouette"],
        save_path=comparison_path,
    )

    final_image = (final_pred.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(final_image).save(OUTPUT_DIR / "final_result.png")

    if gif_frames:
        imageio.mimsave(OUTPUT_DIR / "optimization.gif", gif_frames, duration=0.45)

    save_loss_csv(history)
    save_loss_curve(history)
    save_plain_obj(final_verts, faces)

    print("Done.")
    print(f"Comparison: {comparison_path}")
    print(f"GIF: {OUTPUT_DIR / 'optimization.gif'}")
    print(f"Mesh: {OUTPUT_DIR / 'final_deformed_mesh.obj'}")


def optimize():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    renderer = build_renderer(image_size=IMAGE_SIZE)

    target_silhouettes, cameras, reference_view, source_note = load_target_silhouettes(
        renderer=renderer,
        device=device,
    )
    source_mesh = build_source_mesh(device)

    deform_verts = torch.zeros_like(
        source_mesh.verts_packed(),
        requires_grad=True,
        device=device,
    )
    optimizer = torch.optim.Adam([deform_verts], lr=LEARNING_RATE)

    history: list[dict[str, float]] = []
    gif_frames = []

    print(f"Using device: {device}")
    print(f"Target source: {source_note}")
    print(f"Output dir: {OUTPUT_DIR}")

    for epoch in tqdm(range(NUM_EPOCHS), desc="Optimizing"):
        optimizer.zero_grad()

        current_mesh = source_mesh.offset_verts(deform_verts)
        pred_silhouettes = render_silhouettes(renderer, current_mesh, cameras)

        loss_sil = silhouette_loss(pred_silhouettes, target_silhouettes)
        loss_reg, loss_lap, loss_edge, loss_normal = regularization_loss(
            current_mesh,
            w_lap=W_LAP,
            w_edge=W_EDGE,
            w_normal=W_NORMAL,
        )
        loss = loss_sil + loss_reg

        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch,
                "total": float(loss.item()),
                "silhouette": float(loss_sil.item()),
                "laplacian": float(loss_lap.item()),
                "edge": float(loss_edge.item()),
                "normal": float(loss_normal.item()),
            }
        )

        if epoch in SNAPSHOT_EPOCHS:
            snapshot = save_snapshot(
                target_view=reference_view,
                pred_view=pred_silhouettes[0],
                epoch=epoch,
                total_loss=float(loss.item()),
                silhouette_err=float(loss_sil.item()),
            )
            gif_frames.append(imageio.imread(snapshot))

            print(
                f"Epoch {epoch:03d} | Total: {loss.item():.4f} | "
                f"Silhouette: {loss_sil.item():.4f} | Lap: {loss_lap.item():.4f} | "
                f"Edge: {loss_edge.item():.4f} | Normal: {loss_normal.item():.4f}"
            )

    final_mesh = source_mesh.offset_verts(deform_verts)
    final_pred = render_silhouettes(renderer, final_mesh, cameras)[0]

    comparison_path = OUTPUT_DIR / "final_comparison.png"
    save_snapshot(
        target_view=reference_view,
        pred_view=final_pred,
        epoch=NUM_EPOCHS - 1,
        total_loss=history[-1]["total"],
        silhouette_err=history[-1]["silhouette"],
        save_path=comparison_path,
    )

    final_image = (final_pred.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(final_image, mode="L").save(OUTPUT_DIR / "final_result.png")

    if gif_frames:
        imageio.mimsave(OUTPUT_DIR / "optimization.gif", gif_frames, duration=0.45)

    save_loss_csv(history)
    save_loss_curve(history)
    save_final_mesh(final_mesh)

    print("Done.")
    print(f"Comparison: {comparison_path}")
    print(f"GIF: {OUTPUT_DIR / 'optimization.gif'}")
    print(f"Mesh: {OUTPUT_DIR / 'final_deformed_mesh.obj'}")


def main() -> None:
    if has_pytorch3d():
        optimize()
        return

    optimize_fallback()


if __name__ == "__main__":
    main()
