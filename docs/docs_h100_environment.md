# H100 训练环境

本文档说明如何使用独立的 `environment_h100.yml` 创建适配本地 H100 的 CUDA 12 / PyTorch 2 环境；原始 `environment.yml` 保持为仓库默认环境。

本仓库的两个 CUDA 扩展 `diff-gaussian-rasterization` 和 `simple-knn` 在 `setup.py` 中直接导入 `torch`。为了让 `conda env create -f environment_h100.yml` 稳定成功，`environment_h100.yml` 只负责创建 PyTorch/CUDA/通用 Python 依赖；创建成功后再用 `pip install --no-build-isolation` 安装本地 CUDA 扩展。

## 环境名称

```text
half_gaussian_splatting_h100
```

## 创建环境

H100 对应 CUDA compute capability 为 `sm_90`。创建环境前建议显式设置 CUDA 扩展编译架构：

```bash
export TORCH_CUDA_ARCH_LIST="9.0"
conda env create -f environment_h100.yml
```

如果环境已经存在，可以先删除后重建：

```bash
conda env remove -n half_gaussian_splatting_h100
export TORCH_CUDA_ARCH_LIST="9.0"
conda env create -f environment_h100.yml
```

## 安装本地 CUDA 扩展

```bash
conda activate half_gaussian_splatting_h100
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include/cccl:${CPATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./submodules/simple-knn
```

## 验证环境

```bash
conda activate half_gaussian_splatting_h100

python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn._C
import lpips

print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
print("diff_gaussian_rasterization OK")
print("simple_knn OK")
print("lpips OK")
PY
```

## 训练命令

```bash
conda activate half_gaussian_splatting_h100
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python train.py -s t2data/Train_colmap --eval -m output/Train_3dhgs_h100
```

注意：不要省略 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`。在当前机器上，裸跑 `python train.py ...` 时 PyTorch 的默认 `cuda:0` 可能映射到 L20；加上这两个环境变量后，只暴露物理 GPU 0，也就是 `NVIDIA H100 PCIe`。
