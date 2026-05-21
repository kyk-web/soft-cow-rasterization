from __future__ import annotations

import os
import numpy as np

try:
    from pytorch3d.renderer import (
        BlendParams,
        MeshRasterizer,
        MeshRenderer,
        RasterizationSettings,
        SoftSilhouetteShader,
    )
except ImportError:
    BlendParams = None
    MeshRasterizer = None
    MeshRenderer = None
    RasterizationSettings = None
    SoftSilhouetteShader = None


def require_pytorch3d_renderer():
    if (
        BlendParams is None
        or MeshRasterizer is None
        or MeshRenderer is None
        or RasterizationSettings is None
        or SoftSilhouetteShader is None
    ):
        raise RuntimeError(
            "当前环境缺少 pytorch3d.renderer，无法构建 SoftSilhouetteShader 渲染器。"
        )


def build_silhouette_shader():
    require_pytorch3d_renderer()

    blend_params = BlendParams(
        sigma=1e-4,
        gamma=1e-4,
        background_color=(0.0, 0.0, 0.0),
    )
    return SoftSilhouetteShader(blend_params=blend_params)


def build_renderer(image_size: int):
    require_pytorch3d_renderer()

    blend_params = BlendParams(
        sigma=1e-4,
        gamma=1e-4,
        background_color=(0.0, 0.0, 0.0),
    )
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=int(os.getenv("SCR_FACES_PER_PIXEL", "15")),
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )
