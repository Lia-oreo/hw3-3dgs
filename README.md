# HW3: 基于 3DGS 与 AIGC 的多源资产生成与真实场景融合

## 项目简介

本项目实现了一个"全链路"的 3D 视觉实验，包含三种不同技术路径的 3D 物体生成、背景场景重建，以及场景融合渲染。

### 技术路径

| 物体 | 方法 | 输入 | 技术栈 |
|------|------|------|--------|
| 物体 A | 真实多视角重建 | 58张环绕照片 | COLMAP + 3DGS |
| 物体 B | 文本到3D生成 | 文本Prompt | SD 2.1 + 3DGS |
| 物体 C | 单图到3D生成 | 单张照片 | SD img2img + 3DGS |
| 背景 | 开源数据集重建 | Mip-NeRF 360 Garden | 3DGS |

### 主要结果

- 物体 A（饮料杯）：PSNR = 41.03 dB
- 物体 B（红苹果）：PSNR = 41.03 dB
- 物体 C（牛奶盒）：PSNR = 16.12 dB
- 融合漫游视频：120帧 @ 30fps

---

## 环境配置

### 硬件要求

- GPU：NVIDIA RTX 3090（24GB VRAM）或更高
- 内存：32GB+
- 存储：50GB+

### 软件环境

| 组件 | 版本 |
|------|------|
| PyTorch | 2.0.0+cu118 |
| Python | 3.8+ |
| CUDA | 11.8 |
| COLMAP | 3.9 |

### 安装依赖

```bash
# 创建 conda 环境
conda create -n gs python=3.8 -y
conda activate gs

# 安装 PyTorch
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# 克隆 3DGS 并安装
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install plyfile tqdm
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# 安装 diffusers（用于物体 B/C 生成）
pip install diffusers==0.20.2 transformers accelerate
pip install "numpy<2"
```

### COLMAP 安装（Ubuntu）

```bash
sudo apt-get install colmap
```

---

## 数据准备

### 物体 A（真实物体）

1. 用手机拍摄一个真实物体的环绕视频或多视角照片（50-100张）
2. 将照片 resize 到合适大小（建议 800px）
3. 放入 `object_a/images/` 目录

```bash
# 使用 ImageMagick resize
mogrify -resize 800x800 object_a/images/*.jpg
```

### 背景场景

下载 Mip-NeRF 360 数据集：

```bash
mkdir -p datasets/garden
wget http://storage.googleapis.com/gresearch/refreal360/garden.zip
unzip garden.zip -d datasets/garden/
```

---

## 运行流程

### Step 1: COLMAP 位姿提取（物体 A）

```bash
# 特征提取
QT_QPA_PLATFORM=offscreen colmap feature_extractor \
  --database_path object_a/database.db \
  --image_path object_a/images \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 0

# 特征匹配
QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher \
  --database_path object_a/database.db \
  --SiftMatching.use_gpu 0

# 稀疏重建
mkdir -p object_a/sparse
QT_QPA_PLATFORM=offscreen colmap mapper \
  --database_path object_a/database.db \
  --image_path object_a/images \
  --output_path object_a/sparse \
  --Mapper.init_min_num_inliers 30
```

### Step 2: 多视角图像生成（物体 B/C）

```bash
# 物体 B：文本生成
python scripts/gen_views_b.py

# 物体 C：单图生成
python scripts/gen_views_c.py
```

### Step 3: 构造 COLMAP 格式数据（物体 B/C）

```bash
python scripts/create_fake_colmap.py  # 在 object_b/ 和 object_c/ 目录分别运行
```

### Step 4: 3DGS 训练

```bash
# 物体 A
python gaussian-splatting/train.py \
  -s object_a \
  -m object_a/output \
  --iterations 7000

# 物体 B
python gaussian-splatting/train.py \
  -s object_b \
  -m object_b/output \
  --iterations 7000

# 物体 C（换端口避免冲突）
python gaussian-splatting/train.py \
  -s object_c \
  -m object_c/output \
  --iterations 7000 \
  --port 6010

# 背景场景
python gaussian-splatting/train.py \
  -s datasets/garden \
  -m datasets/garden/output \
  --iterations 7000
```

### Step 5: 渲染

```bash
# 渲染训练视角图像
python gaussian-splatting/render.py -m object_a/output --skip_test
python gaussian-splatting/render.py -m object_b/output --skip_test
python gaussian-splatting/render.py -m object_c/output --skip_test
python gaussian-splatting/render.py -m datasets/garden/output --skip_test
```

### Step 6: 场景融合

```bash
# 合并点云
python scripts/merge_ply.py

# 生成漫游视频
python scripts/render_video.py \
  --model_path merge \
  --output_path merge/video \
  --num_frames 120
```

---

## 项目结构

```
.
├── README.md
├── scripts/
│   ├── create_fake_colmap.py    # 构造 COLMAP 格式数据
│   ├── gen_views_b.py           # 物体 B 多视角生成
│   ├── gen_views_c.py           # 物体 C 多视角生成
│   ├── merge_ply.py             # 点云合并
│   └── render_video.py          # 视频渲染
├── object_a/                    # 物体 A 数据与模型
│   ├── images/
│   ├── sparse/
│   └── output/
├── object_b/                    # 物体 B 数据与模型
│   ├── views/
│   ├── sparse/
│   └── output/
├── object_c/                    # 物体 C 数据与模型
│   ├── views/
│   ├── sparse/
│   └── output/
├── datasets/garden/             # 背景场景
│   └── output/
└── merge/                       # 融合场景
    └── video/
```

---

## 模型权重

训练好的模型权重可通过以下链接下载：

- **百度网盘**：[链接]（提取码：xxxx）
- **Google Drive**：[链接]

模型文件说明：
- `object_a/output/point_cloud/iteration_7000/point_cloud.ply`
- `object_b/output/point_cloud/iteration_7000/point_cloud.ply`
- `object_c/output/point_cloud/iteration_7000/point_cloud.ply`
- `datasets/garden/output/point_cloud/iteration_7000/point_cloud.ply`
- `merge/point_cloud/iteration_7000/point_cloud.ply`（融合模型）

---

## 实验结果

### 定量指标

| 物体 | 迭代次数 | 训练时间 | PSNR (dB) | L1 Loss |
|------|----------|----------|-----------|---------|
| 物体 A | 7000 | ~1 min | 41.03 | 0.0063 |
| 物体 B | 7000 | ~1 min | 41.03 | 0.0063 |
| 物体 C | 7000 | ~1 min | 16.12 | 0.1238 |

### 渲染示例

| 物体 A | 物体 B | 背景 |
|--------|--------|------|
| ![A](results/object_a.png) | ![B](results/object_b.png) | ![BG](results/garden.png) |

---

## 已知局限性

1. **物体 C 效果差**：SD img2img 多视角一致性不足，建议使用 Zero123/SyncDreamer
2. **物体 B 替代方案**：因环境限制使用 SD 2.1 base 替代 threestudio + SDS Loss
3. **融合简化**：未进行精细尺度对齐和遮挡处理

---

## 参考

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [COLMAP](https://colmap.github.io/)
- [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refreal360/)

---
