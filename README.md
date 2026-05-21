# 基于 PyTorch3D Soft Rasterization 的球体到牛形轮廓可微优化实验

## 一、实验目标

本实验围绕可微渲染中的软光栅化展开，目标是将一个初始球体网格通过梯度优化逐步形变为牛的轮廓。整个过程直接优化网格顶点偏移量，使渲染结果与目标牛模型的剪影尽可能接近，同时利用网格正则项保持形变平滑稳定。

本实验主要目标如下：

1. 理解软光栅化在可微渲染中的作用。
2. 掌握如何使用二维剪影监督优化三维网格。
3. 理解拉普拉斯平滑、边长一致性和法线一致性在网格优化中的意义。
4. 输出可直接写入实验报告的对比图、GIF、Loss 曲线和最终网格。

---

## 二、实验环境

本项目仓库中同时保留了兼容版回退逻辑，但本次实验报告使用的是**真实 PyTorch3D 版本**的运行结果。

| 项目 | 内容 |
| --- | --- |
| 操作系统 | Windows 10 |
| IDE | Trae |
| 环境管理 | `uv` |
| 真实运行环境 | `.venv310` |
| Python 版本 | Python 3.10.20 |
| 主要依赖 | `torch==2.4.1`、`torchvision==0.19.1`、`pytorch3d==0.7.9`、`matplotlib`、`imageio`、`pillow`、`tqdm` |
| 目标模型 | `assets/cow.obj` |
| 运行设备 | CPU |

说明：

1. 由于 Windows 下 `PyTorch3D` 与 `Python 3.11` / 最新 `torch` 轮子兼容性较差，真实运行单独使用了 `.venv310`。
2. `cow.obj` 没有配套 `cow.mtl` 不影响本实验，因为我们只使用剪影渲染，不使用纹理贴图。

---

## 三、实验原理

### 3.1 软光栅化

传统光栅化会将像素是否落在三角形内部视为离散判断，这会导致梯度难以反向传播。软光栅化则将边界附近的覆盖关系变为连续概率，使渲染过程可导。

本实验使用 `PyTorch3D` 中的 `SoftSilhouetteShader` 生成软剪影图，核心思想是通过模糊边界让顶点移动能够影响像素值，从而把轮廓误差传回到三维网格顶点。

### 3.2 轮廓监督

设目标牛网格在相机视角下渲染得到的剪影为 `target_silhouette`，当前可变形球体渲染得到的剪影为 `pred_silhouette`，则轮廓损失为：

```text
L_silhouette = MSE(pred_silhouette, target_silhouette)
```

该项负责引导球体整体轮廓向牛轮廓靠近。

### 3.3 网格正则化

如果只最小化剪影误差，网格容易出现尖刺、拉伸和局部塌陷，因此实验中加入三类几何正则项：

1. 拉普拉斯平滑 `L_lap`：约束局部邻域平滑，防止表面过于尖锐。
2. 边长一致性 `L_edge`：抑制三角形边长过度变化。
3. 法线一致性 `L_normal`：鼓励相邻三角面的法线方向接近。

总损失函数为：

```text
L_total = L_silhouette + w_lap L_lap + w_edge L_edge + w_normal L_normal
```

本次真实运行使用的权重为：

```text
w_lap = 0.02
w_edge = 0.20
w_normal = 0.01
```

---

## 四、项目结构

当前项目的核心结构如下：

```text
soft-cow-rasterization/
├─ assets/
│  └─ cow.obj
├─ data/
│  └─ target.png
├─ outputs/
│  ├─ epoch_000.png
│  ├─ epoch_025.png
│  ├─ epoch_050.png
│  ├─ epoch_075.png
│  ├─ epoch_100.png
│  ├─ epoch_150.png
│  ├─ epoch_200.png
│  ├─ epoch_250.png
│  ├─ epoch_299.png
│  ├─ final_comparison.png
│  ├─ final_result.png
│  ├─ final_deformed_mesh.obj
│  ├─ loss_curve.png
│  ├─ loss_log.csv
│  └─ optimization.gif
├─ src/
│  └─ soft_cow_rasterization/
│     ├─ __init__.py
│     ├─ losses.py
│     ├─ main.py
│     └─ shader.py
├─ pyproject.toml
├─ uv.lock
└─ README.md
```

### 4.1 代码模块说明

| 文件 | 作用 |
| --- | --- |
| `main.py` | 主训练流程、目标加载、渲染、优化、保存输出 |
| `shader.py` | 构建 `SoftSilhouetteShader` 与 `MeshRenderer` |
| `losses.py` | 轮廓损失与网格正则化损失 |
| `assets/cow.obj` | 目标牛模型 |
| `outputs/*` | 本次真实运行生成的实验结果 |

---

## 五、实现流程

### 5.1 加载目标模型并生成参考剪影

程序首先读取 `assets/cow.obj`，然后使用 `PyTorch3D` 渲染出目标牛网格的剪影图。当前真实实验配置使用单前视角：

```text
elev = 0.0
azim = 180.0
dist = 2.7
```

### 5.2 构造初始球体网格

源模型不是直接读取球体文件，而是通过 `PyTorch3D` 的 `ico_sphere` 生成初始球网格，再对其顶点做归一化。当前真实实验使用：

```text
ico level = 3
```

优化变量为每个顶点的偏移量：

```text
verts = base_verts + deform_verts
```

其中 `deform_verts` 设置为 `requires_grad=True`。

### 5.3 构建软剪影渲染器

真实实验使用 `MeshRenderer + MeshRasterizer + SoftSilhouetteShader` 组成完整渲染链路。关键参数包括：

```text
image_size = 96
faces_per_pixel = 15
sigma = 1e-4
gamma = 1e-4
```

### 5.4 进行可微优化

每次迭代的流程如下：

```text
球体网格 + 顶点偏移
        ↓
SoftSilhouetteShader 渲染当前剪影
        ↓
与目标牛剪影计算 MSE 轮廓误差
        ↓
加入拉普拉斯 / 边长 / 法线正则项
        ↓
Adam 更新 deform_verts
        ↓
保存关键 epoch 图像与 Loss 日志
```

优化器使用：

```text
Adam
lr = 0.20
epochs = 300
```

---

## 六、运行方式

如果要复现实验报告中的真实结果，可以在仓库根目录执行：

```powershell
$env:PYTHONPATH='src'
.\.venv310\Scripts\python.exe -c "from soft_cow_rasterization.main import main; main()"
```

---

## 七、实验结果

### 7.1 优化过程动态图

下图展示了真实 PyTorch3D 版本下，球体网格逐步向目标牛剪影逼近的过程：

![优化过程](outputs/optimization.gif)

### 7.2 最终轮廓对比图

下图为本次真实运行的最终对比结果。左侧是由 `cow.obj` 渲染得到的目标剪影，右侧是第 `299/300` 轮优化结果。

![最终轮廓对比](outputs/final_comparison.png)

### 7.3 Loss 曲线

优化过程中总损失与剪影损失的变化如下图所示：

![Loss 曲线](outputs/loss_curve.png)

### 7.4 中间过程截图

为了便于实验报告展示，程序还额外保存了多个关键 epoch 的对比图，例如：

| Epoch | 图片 |
| --- | --- |
| 0 | ![](outputs/epoch_000.png) |
| 75 | ![](outputs/epoch_075.png) |
| 150 | ![](outputs/epoch_150.png) |
| 299 | ![](outputs/epoch_299.png) |

---

## 八、真实运行结果摘要

本次真实 PyTorch3D 运行的最终结果如下：

| 指标 | 数值 |
| --- | --- |
| 最终 epoch | 299 / 300 |
| Total Loss | 0.0318 |
| Silhouette Loss | 0.0108 |
| 最终对比图 | `outputs/final_comparison.png` |
| 优化 GIF | `outputs/optimization.gif` |
| 最终网格 | `outputs/final_deformed_mesh.obj` |

`loss_log.csv` 中最后几轮记录如下：

```text
295,0.02802804298698902,0.010475652292370796,0.09403694421052933,0.05593477934598923,0.4484695792198181
296,0.029290400445461273,0.010520456358790398,0.0996367409825325,0.060687556862831116,0.46396973729133606
297,0.03560475632548332,0.0153660224750638,0.10491617023944855,0.0673513412475586,0.4670140743255615
298,0.03192552179098129,0.011219998821616173,0.10603085905313492,0.07027704268693924,0.452949583530426
299,0.0318484902381897,0.0107694361358881,0.1062241643667221,0.07232854515314102,0.44888603687286377
```

---

## 九、输出文件说明

本次真实运行后，`outputs` 目录中保留了以下可直接用于实验报告或提交的文件：

| 文件 | 说明 |
| --- | --- |
| `final_comparison.png` | 最终目标剪影与优化结果对比图 |
| `optimization.gif` | 优化过程动态图 |
| `loss_curve.png` | Loss 曲线图 |
| `loss_log.csv` | 全部训练轮次的损失记录 |
| `epoch_*.png` | 关键轮次中间过程截图 |
| `final_result.png` | 最终预测剪影单图 |
| `final_deformed_mesh.obj` | 最终变形后的网格 |

---

## 十、实验总结

本实验已经使用真实的 `PyTorch3D` 渲染链路完成了球体网格到牛形轮廓的可微优化过程。实验中通过 `SoftSilhouetteShader` 将剪影渲染转化为可导过程，再通过剪影误差和网格正则化共同优化顶点偏移量，最终得到一组可视化完整、可写入实验报告的结果文件。

从实验现象可以看出：

1. 软剪影监督能够有效驱动三维球体向目标轮廓形变。
2. 正则化项对抑制网格畸变非常关键。
3. 在 CPU 环境下，适当降低分辨率、减少视角数、控制球体细分级别，可以显著缩短真实版实验时间。

---

## 十一、参考资料

1. PyTorch3D 官方文档
2. PyTorch 官方文档
3. 课程 PPT 中关于软光栅化和网格优化的相关内容
4. Differentiable Rendering 与 Soft Rasterization 相关资料
