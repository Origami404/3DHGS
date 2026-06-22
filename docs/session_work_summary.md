# Conda 环境创建修复记录（2026-06-19 至 2026-06-20）

## 1. 目标与初始失败

- 用户要求想办法跑成功：

```bash
conda env create --file environment.yml
```

- 初始 `environment.yml` 使用：
  - Python 3.7.13
  - PyTorch 1.12.1 + CUDA 11.6
  - `cudatoolkit=11.6`
  - pip 安装本地 CUDA/C++ 扩展：
    - `submodules/diff-gaussian-rasterization`
    - `submodules/simple-knn`
- 第一次创建失败在 pip 阶段，`torch` import 报：

```text
undefined symbol: iJIT_NotifyEvent
```

- 根因：PyTorch 1.12.1 与过新的 MKL 不兼容。
- 修复：在 `environment.yml` 中固定：

```yaml
- mkl<2024
```

## 2. CUDA 编译环境问题

- 继续创建后，pip 编译本地扩展失败。
- 发现机器全局 CUDA 是：

```text
/usr/local/cuda -> CUDA 13.2
```

- 但当前 PyTorch 是 CUDA 11.6 版本，不能使用全局 CUDA 13.2 编译扩展。
- 明确要求：不能影响全局 CUDA 配置。
- 采取方案：只在 conda env 内补充 CUDA 11.6 编译工具，不修改 `/usr/local/cuda`、alternatives 或系统 PATH。
- 在 `environment.yml` 中加入：

```yaml
channels:
  - nvidia
  - pytorch
  - conda-forge
  - defaults

dependencies:
  - cuda-cudart-dev=11.6.55
  - cuda-nvcc=11.6.55
  - cudatoolkit=11.6
  - ninja
```

- 验证 env 内 `nvcc` 路径为：

```text
/home/s_liangtao/miniforge3/envs/half_gaussian_splatting/bin/nvcc
```

- 验证版本为：

```text
Cuda compilation tools, release 11.6, V11.6.55
```

## 3. 编译器与头文件问题

- 使用 env 内 CUDA 后，继续遇到系统 GCC 13 与 CUDA 11.6 不兼容的问题。
- 讨论确认：`environment.yml` 应提供完整、隔离、可复现的编译工具链，而不是依赖系统 GCC。
- 先尝试 env 内 GCC/G++ 11，确认激活后编译确实使用 env 内编译器：

```text
CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
```

- 之后遇到 `Python.h` 间接包含 `crypt.h`，但 conda sysroot 下找不到：

```text
fatal error: crypt.h: No such file or directory
```

- 解释了 `libxcrypt` 不是业务加密依赖，而是 Python C 扩展编译所需的系统级头文件/库。
- 在 `environment.yml` 中加入：

```yaml
- libxcrypt
```

## 4. GCC 11 不适配并降级到 GCC 10

- 使用 env 内 GCC 11.4 后，CUDA 11.6 编译 `simple-knn` 时遇到：

```text
std_function.h: error: parameter packs not expanded with '...'
```

- 判断为 CUDA 11.6 + GCC 11.4/libstdc++ 组合的实际兼容性问题。
- 将 env 内编译器从 GCC/G++ 11 改为 GCC/G++ 10：

```yaml
- gcc_linux-64=10
- gxx_linux-64=10
```

- 验证 env 内编译器版本：

```text
x86_64-conda-linux-gnu-g++ (conda-forge gcc 10.4.0-19) 10.4.0
```

- 用 GCC 10 成功编译并安装两个本地扩展：
  - `simple_knn`
  - `diff_gaussian_rasterization`

## 5. Pillow/libtiff ABI 问题

- `conda env create` 成功后做导入验证，发现 `lpips` -> `torchvision` -> `PIL` 导入失败：

```text
ImportError: libtiff.so.5: cannot open shared object file: No such file or directory
```

- 检查发现 `pillow` 来自 defaults，期望 `libtiff.so.5`；但 `libtiff` 被 conda-forge 解成了 `libtiff.so.6`。
- 修复：固定 Pillow 与 libtiff 到兼容版本：

```yaml
- pillow=9.2.0
- libtiff=4.4.0
```

- 验证后 `PIL`、`torchvision`、`lpips` 均可导入。

## 6. 最终验证结果

- 最终从零执行成功：

```bash
conda env remove -n half_gaussian_splatting -y
conda env create --file environment.yml
```

- pip 阶段成功构建并安装：

```text
Successfully built diff_gaussian_rasterization simple_knn
Successfully installed diff_gaussian_rasterization-0.0.0 lpips-0.1.4 scipy-1.7.3 simple_knn-0.0.0
```

- 最终导入验证全部通过：

```text
python import ok
torch 1.12.1 cuda 11.6 cuda_available True
torchvision 0.13.1
PIL 9.2.0
scipy 1.7.3
simple_knn simple_knn._C
diff_gaussian_rasterization ok
```

- 最终工具链验证：

```text
/home/s_liangtao/miniforge3/envs/half_gaussian_splatting/bin/nvcc
Cuda compilation tools, release 11.6, V11.6.55
x86_64-conda-linux-gnu-g++ (conda-forge gcc 10.4.0-19) 10.4.0
```

- 确认没有修改全局 CUDA 或系统 GCC；所有 CUDA/GCC 编译工具均来自 conda env。

# 本轮对话工作总结

## 1. 阅读仓库与论文

- 阅读了 `README.md`、训练入口、渲染入口、数据读取逻辑和 CUDA rasterizer 相关实现。
- 阅读并解析了本地论文 `论文.pdf`。
- 总结了 3D-HGS 的核心思路：
  - 在 3D Gaussian 基础上加入切分平面 normal。
  - 每个 Gaussian 被分成两个 half-Gaussian。
  - 两个 half-Gaussian 使用不同 opacity，以更好表达边界/高频不连续区域。
- 对照代码指出关键实现位置：
  - `scene/gaussian_model.py`
  - `gaussian_renderer/__init__.py`
  - `submodules/diff-gaussian-rasterization/`

## 2. 确认数据集格式

- 阅读 `scene/dataset_readers.py` 和 `scene/__init__.py`。
- 确认仓库支持两类输入：
  - COLMAP/3D-GS 格式：`images/` + `sparse/0/*.bin|*.txt`
  - Blender synthetic 格式：`transforms_train.json`
- 说明 Tanks&Temples 复现实验应使用 COLMAP/3D-GS 格式。
- 检查了用户已有的 `t2data/image_sets/*.zip`，发现它们实际是 Google 登录 HTML，不是有效 zip。

## 3. 下载并准备 Tanks&Temples Train 场景

- 找到 3D-GS 官方预处理包：`tandt_db.zip`。
- 下载到：`t2data/downloads/tandt_db.zip`。
- 从压缩包中只抽取 `tandt/train` 场景。
- 整理为仓库可直接读取的目录：

```text
t2data/Train_colmap/
  images/
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin
```

- 使用仓库读取逻辑验证：
  - 图片数量：`301`
  - train cameras：`263`
  - test cameras：`38`
  - 初始点云：`182686` points
- 生成了 `points3D.ply`。

## 4. 编写数据准备文档

- 新增/整理了数据准备说明文档。
- 文档记录了：
  - 下载 `tandt_db.zip`
  - 检查 zip 有效性
  - 抽取 `tandt/train`
  - 验证 COLMAP 数据读取
  - 训练/测试命令
- 文档路径：`docs/docs_prepare_tandt_train.md`

## 5. 训练 Train 场景并解释输出

- 用户运行了训练命令：

```bash
python train.py -s t2data/Train_colmap --eval -m output/Train_3dhgs
```

- 解释了训练输入：
  - Tanks&Temples 的 Train 火车场景。
  - 多视角拍摄的 Western Pacific 713 火车。
- 解释了训练输出：
  - `output/Train_3dhgs/input.ply`
  - `output/Train_3dhgs/cameras.json`
  - `output/Train_3dhgs/cfg_args`
  - `output/Train_3dhgs/point_cloud/iteration_30000/point_cloud.ply`
  - `output/Train_3dhgs/tensor_data.npy`
- 检查了最终模型规模：
  - 初始点云：`182686`
  - 训练后 Gaussian 数量：约 `1182310`

## 6. 初版输入 preview 和后续纠正

- 最初写了一个按文件序号抽样的输入图片 preview 文档。
- 后续用户指出该方式没有根据相机位姿，确实不严谨。
- 承认问题并修正方向：
  - 不再把“按文件名抽样”当作“视角 preview”。
  - 改成基于 COLMAP `images.bin` 的真实相机位姿。

## 7. 增加训练 loss 保存

- 修改了 `train.py`，为后续训练增加 loss 记录：
  - `train_loss.csv`
  - `train_loss_curve.png`
- 每个 iteration 写入：
  - `iteration`
  - `l1_loss`
  - `total_loss`
  - `ema_loss`
- 每 1000 iter 和最后一次自动刷新 loss 曲线图。
- 用 `py_compile` 验证了语法。

## 8. 清理 Git 中的 pycache

- 执行了 `git rm --cached`，从索引移除所有已跟踪的 `__pycache__` / `.pyc` 文件。
- 保留本地文件，但让它们在后续提交中从 Git 仓库删除。

## 9. 创建 H100 环境分支并更新 conda 环境

- 新建分支：`h100-conda-env`。
- 初始尝试直接改 `environment.yml` 为 H100 环境。
- 多次运行 `conda env create`，遇到过：
  - 网络/SSL 下载错误。
  - pip build isolation 导致本地 CUDA 扩展无法 import torch。
  - CUDA 扩展编译时缺少 `cuda/std/type_traits`。
- 按用户要求，网络错误先直接重试，三次仍失败再考虑其他方案。
- 最终调整策略：
  - `conda env create` 只负责 PyTorch/CUDA/通用依赖。
  - 本地 CUDA 扩展单独用 `pip install --no-build-isolation` 安装。

## 10. 拆分 H100 环境文件并提交

- 根据用户要求，将 H100 环境从默认环境拆开。
- 恢复原始 `environment.yml`：
  - `half_gaussian_splatting`
  - CUDA 11.6
  - PyTorch 1.12.1
- 新增 H100 专用文件：
  - `environment_h100.yml`
  - 环境名：`half_gaussian_splatting_h100`
  - CUDA 12.4
  - PyTorch 2.5.1
- 新增/修正文档：`docs/docs_h100_environment.md`
- 切回 `main` 并提交：

```text
45dc8fe Add H100 conda environment
```

## 11. 确认 H100 而不是 L20 被使用

- 检查 `nvidia-smi`，发现物理 GPU：
  - GPU 0-3：NVIDIA H100 PCIe
  - GPU 4-5：NVIDIA L20
- 发现 PyTorch 默认枚举顺序会把 L20 放在 `cuda:0`。
- 确认必须使用：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
```

- 这样 PyTorch 只看到物理 GPU 0，即 H100。
- 验证输出：

```text
count 1
device0 NVIDIA H100 PCIe (9, 0)
torch 2.5.1 cuda 12.4
```

- 在 H100 环境中安装了本地 CUDA 扩展：
  - `diff_gaussian_rasterization`
  - `simple_knn`
- 安装时需要：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include/cccl:${CPATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"
```

- 用 H100 成功跑通 1-iteration 训练冒烟：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
conda run -n half_gaussian_splatting_h100 \
python train.py -s t2data/Train_colmap --eval \
  -m output/Train_3dhgs_h100_smoke \
  --iterations 1 --save_iterations 1 --test_iterations 1
```

## 12. 确认图片与视角的对应关系

- 解释了 `images/` 中图片本身不包含视角。
- 每张图片对应的相机位姿来自：

```text
t2data/Train_colmap/sparse/0/images.bin
```

- `images.bin` 每条记录包含：
  - `image_id`
  - `qvec`
  - `tvec`
  - `camera_id`
  - `name`
  - 2D-3D feature correspondences
- 通过图片名匹配，例如 `name == "00001.jpg"`。
- 抽取了示例相机信息：

```text
00001.jpg
image_id: 1
camera_id: 1
camera_center_world: [-3.008989 -0.110865 -3.752764]
forward_world: [ 0.477066 -0.057587  0.876979]
```

## 13. 实现基于相机位姿的渲染 preview

- 新增脚本：`scripts/make_pose_render_preview.py`
- 作用：根据某张原始图片在 COLMAP 中的相机位置，生成四宫格大图：
  1. 原始输入图。
  2. 同 COLMAP 相机位姿的渲染图。
  3. 固定相机中心、向左轻微偏转的渲染图。
  4. 固定相机中心、向右轻微偏转的渲染图。
- 大图底部写入：
  - image name
  - image_id
  - camera_id
  - camera_center_world
  - forward_world
- 使用 `output/Train_3dhgs_h100` 的训练结果生成了示例：

```text
output/Train_3dhgs_h100/previews/pose_render_preview_00001.jpg
```

- 修正了 preview 文档：`docs/docs_make_input_previews.md`
- 提交：

```text
17fc624 Add pose-aware render preview tool
```

## 14. 当前重要命令

### H100 正式训练命令

```bash
conda activate half_gaussian_splatting_h100
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python train.py -s t2data/Train_colmap --eval -m output/Train_3dhgs_h100
```

### 生成基于位姿的 preview

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
conda run -n half_gaussian_splatting_h100 \
python scripts/make_pose_render_preview.py \
  -s t2data/Train_colmap \
  -m output/Train_3dhgs_h100 \
  --image 00001.jpg \
  --iteration 30000
```

## 15. 已提交的关键提交

```text
45dc8fe Add H100 conda environment
17fc624 Add pose-aware render preview tool
```

## 16. 未提交说明

- 本文档位于 `.local/session_work_summary.md`。
- `.local/` 不应提交到 Git。
