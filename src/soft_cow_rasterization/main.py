from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import trange


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_uv_sphere(rows: int = 6, cols: int = 12, radius: float = 0.62):
    verts = []
    for i in range(rows + 1):
        theta = math.pi * i / rows
        z = radius * math.cos(theta)
        r = radius * math.sin(theta)
        for j in range(cols):
            phi = 2 * math.pi * j / cols
            verts.append([r * math.cos(phi), r * math.sin(phi), z])

    faces = []
    for i in range(rows):
        for j in range(cols):
            a = i * cols + j
            b = i * cols + (j + 1) % cols
            c = (i + 1) * cols + j
            d = (i + 1) * cols + (j + 1) % cols
            if i != 0:
                faces.append([a, c, b])
            if i != rows - 1:
                faces.append([b, c, d])

    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def read_obj(path: Path):
    verts, faces = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                ids = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                for k in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[k], ids[k + 1]])

    if not verts or not faces:
        raise ValueError(f"无法从 {path} 读取 OBJ 顶点/面片")

    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def normalize_xy(verts: torch.Tensor, scale: float = 0.78) -> torch.Tensor:
    v = verts.clone()
    xy = v[:, :2]
    center = (xy.max(dim=0).values + xy.min(dim=0).values) / 2
    span = (xy.max(dim=0).values - xy.min(dim=0).values).max().clamp_min(1e-6)
    v[:, :2] = (xy - center) / span * 2 * scale
    v[:, 2] = v[:, 2] - v[:, 2].mean()
    return v


def unique_edges(faces: torch.Tensor) -> torch.Tensor:
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    e = torch.sort(e, dim=1).values
    return torch.unique(e, dim=0)


def adjacent_face_pairs(faces: torch.Tensor) -> torch.Tensor:
    edge_to_face = {}
    pairs = []
    for fi, tri in enumerate(faces.cpu().tolist()):
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = tuple(sorted((a, b)))
            if key in edge_to_face:
                pairs.append([edge_to_face[key], fi])
            else:
                edge_to_face[key] = fi

    return torch.tensor(pairs, dtype=torch.long) if pairs else torch.empty((0, 2), dtype=torch.long)


def face_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = verts[faces]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    return n / (n.norm(dim=1, keepdim=True) + 1e-8)


def laplacian_smooth_loss(deform: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return ((deform[edges[:, 0]] - deform[edges[:, 1]]) ** 2).sum(dim=1).mean()


def edge_length_loss(verts: torch.Tensor, edges: torch.Tensor, init_len: torch.Tensor) -> torch.Tensor:
    now = (verts[edges[:, 0]] - verts[edges[:, 1]]).norm(dim=1)
    return ((now - init_len) ** 2).mean()


def normal_consistency_loss(verts: torch.Tensor, faces: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return verts.sum() * 0

    n = face_normals(verts, faces)
    dot = (n[pairs[:, 0]] * n[pairs[:, 1]]).sum(dim=1).clamp(-1, 1)
    return (1 - dot).mean()


def pixel_grid(size: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(1 - 1 / size, -1 + 1 / size, size, device=device)
    x = torch.linspace(-1 + 1 / size, 1 - 1 / size, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def exact_soft_silhouette(
    verts: torch.Tensor,
    faces: torch.Tensor,
    size: int = 96,
    sigma: float = 0.02,
    chunk: int = 128,
) -> torch.Tensor:
    pts = pixel_grid(size, verts.device)
    tri = verts[faces][:, :, :2]
    sil = torch.zeros(pts.shape[0], device=verts.device)

    def cross2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for start in range(0, tri.shape[0], chunk):
        t = tri[start : start + chunk]
        v0, v1, v2 = t[:, 0], t[:, 1], t[:, 2]

        area = cross2(v1 - v0, v2 - v0)
        orient = torch.where(area >= 0, 1.0, -1.0).view(-1, 1)
        p = pts.unsqueeze(0)

        def edge_sd(a, b):
            ab = (b - a).unsqueeze(1)
            ap = p - a.unsqueeze(1)
            return orient * cross2(ab, ap) / (ab.norm(dim=-1) + 1e-8)

        d = torch.minimum(
            torch.minimum(edge_sd(v0, v1), edge_sd(v1, v2)),
            edge_sd(v2, v0),
        )

        occ = torch.sigmoid(d / sigma)
        sil = torch.maximum(sil, occ.max(dim=0).values)

    return sil.reshape(size, size)


def fast_soft_silhouette(
    verts: torch.Tensor,
    faces: torch.Tensor,
    size: int = 96,
    sigma: float = 0.055,
    chunk: int = 512,
) -> torch.Tensor:
    pts = pixel_grid(size, verts.device)
    tri = verts[faces][:, :, :2]

    bary = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [1 / 3, 1 / 3, 1 / 3],
        ],
        dtype=verts.dtype,
        device=verts.device,
    )

    samples = (tri[:, None, :, :] * bary[None, :, :, None]).sum(dim=2).reshape(-1, 2)
    sil = torch.zeros(pts.shape[0], device=verts.device)

    for start in range(0, samples.shape[0], chunk):
        s = samples[start : start + chunk]
        d2 = ((pts[:, None, :] - s[None, :, :]) ** 2).sum(dim=-1)
        occ = torch.exp(-d2 / (2 * sigma * sigma))
        sil = torch.maximum(sil, occ.max(dim=1).values)

    return sil.reshape(size, size).clamp(0, 1)


def render_silhouette(verts, faces, size, renderer, sigma):
    if renderer == "exact":
        return exact_soft_silhouette(verts, faces, size=size, sigma=sigma)

    return fast_soft_silhouette(verts, faces, size=size, sigma=max(sigma, 0.055))


def load_target_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0

    corners = np.r_[
        arr[:8, :8].ravel(),
        arr[-8:, :8].ravel(),
        arr[:8, -8:].ravel(),
        arr[-8:, -8:].ravel(),
    ]

    if corners.mean() > 0.5:
        arr = 1.0 - arr

    arr = (arr > 0.5).astype(np.float32)
    return torch.tensor(arr, device=device)


def procedural_cow(size: int, device: torch.device) -> torch.Tensor:
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    s = size / 128

    def box(a, b, c, e):
        return [int(a * s), int(b * s), int(c * s), int(e * s)]

    d.ellipse(box(37, 38, 91, 101), fill=255)
    d.ellipse(box(45, 18, 83, 56), fill=255)
    d.ellipse(box(33, 27, 52, 49), fill=255)
    d.ellipse(box(76, 27, 95, 49), fill=255)

    d.polygon(
        [(int(50 * s), int(23 * s)), (int(56 * s), int(8 * s)), (int(61 * s), int(27 * s))],
        fill=255,
    )
    d.polygon(
        [(int(68 * s), int(27 * s)), (int(73 * s), int(8 * s)), (int(79 * s), int(23 * s))],
        fill=255,
    )

    d.rounded_rectangle(box(47, 87, 60, 113), radius=int(5 * s), fill=255)
    d.rounded_rectangle(box(68, 87, 81, 113), radius=int(5 * s), fill=255)
    d.ellipse(box(50, 50, 78, 74), fill=255)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.tensor(arr, device=device)


def make_target(size: int, device: torch.device, renderer: str, sigma: float):
    obj = Path("assets/cow.obj")
    png = Path("assets/target.png")

    if obj.exists():
        v, f = read_obj(obj)
        v = normalize_xy(v).to(device)
        f = f.to(device)
        return render_silhouette(v, f, size, renderer, sigma).detach(), "assets/cow.obj"

    if png.exists():
        return load_target_image(png, size, device), "assets/target.png"

    return procedural_cow(size, device), "程序自动生成的牛形参考图"


def save_panel(target: torch.Tensor, pred: torch.Tensor, out: Path, title: str = "") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(target.detach().cpu(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth Silhouette")

    axes[1].imshow(pred.detach().cpu(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(title or "Optimizing")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_loss_curve(log_rows, out: Path) -> None:
    steps = [r[0] for r in log_rows]
    losses = [r[1] for r in log_rows]

    fig = plt.figure(figsize=(5, 3))
    plt.plot(steps, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Optimization Loss")
    plt.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--renderer", choices=["fast", "exact"], default="fast")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(42)

    device = torch.device(args.device)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    target, target_source = make_target(args.size, device, args.renderer, args.sigma)

    base_verts, faces = create_uv_sphere(rows=6, cols=12, radius=0.62)
    base_verts, faces = base_verts.to(device), faces.to(device)

    edges = unique_edges(faces).to(device)
    pairs = adjacent_face_pairs(faces).to(device)

    init_len = (base_verts[edges[:, 0]] - base_verts[edges[:, 1]]).norm(dim=1).detach()

    deform = torch.zeros_like(base_verts, requires_grad=True)
    opt = torch.optim.Adam([deform], lr=args.lr)

    frames, log_rows = [], []

    print(
        f"device={device}, renderer={args.renderer}, "
        f"target={target_source}, verts={len(base_verts)}, faces={len(faces)}"
    )

    for step in trange(args.iters + 1):
        opt.zero_grad()

        verts = base_verts + deform
        pred = render_silhouette(verts, faces, args.size, args.renderer, args.sigma)

        loss_sil = torch.mean((pred - target) ** 2)
        loss_lap = laplacian_smooth_loss(deform, edges)
        loss_edge = edge_length_loss(verts, edges, init_len)
        loss_norm = normal_consistency_loss(verts, faces, pairs)

        loss = loss_sil + 0.05 * loss_lap + 0.20 * loss_edge + 0.01 * loss_norm

        if step < args.iters:
            loss.backward()
            opt.step()

        if step % 10 == 0 or step == args.iters:
            log_rows.append(
                [
                    step,
                    float(loss.detach()),
                    float(loss_sil.detach()),
                    float(loss_lap.detach()),
                    float(loss_edge.detach()),
                    float(loss_norm.detach()),
                ]
            )

        if step % 50 == 0 or step == args.iters:
            frame_path = out_dir / f"frame_{step:04d}.png"
            save_panel(target, pred, frame_path, title=f"Optimizing... Epoch {step}")
            frames.append(imageio.imread(frame_path))

    with (out_dir / "loss_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total", "silhouette", "laplacian", "edge", "normal"])
        writer.writerows(log_rows)

    imageio.mimsave(out_dir / "optim.gif", frames, duration=0.35)
    save_panel(target, pred, out_dir / "final_comparison.png", title=f"Final Epoch {args.iters}")
    save_loss_curve(log_rows, out_dir / "loss_curve.png")

    print("完成。结果在 outputs/final_comparison.png、outputs/optim.gif、outputs/loss_curve.png")


if __name__ == "__main__":
    main()