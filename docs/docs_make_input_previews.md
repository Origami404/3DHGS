# 基于 COLMAP 相机位姿生成渲染 Preview

本文档记录如何生成真正“按相机位姿”的预览图，而不是简单按文件名抽样图片。

生成结果会把 4 张图拼成一张大图：

1. 原始输入图片。
2. 使用同一个 COLMAP 相机位置和朝向渲染出的图片。
3. 在同一个相机中心上向左轻微偏转视角后的渲染图。
4. 在同一个相机中心上向右轻微偏转视角后的渲染图。

大图底部会写入该图片对应的 COLMAP 相机信息，例如：

```text
00001.jpg
image_id: 1
camera_id: 1
camera_center_world: [-3.008989 -0.110865 -3.752764]
forward_world: [ 0.477066 -0.057587  0.876979]
```

## 数据来源

图片和视角的对应关系来自 COLMAP sparse model：

```text
t2data/Train_colmap/images/00001.jpg
t2data/Train_colmap/sparse/0/images.bin
t2data/Train_colmap/sparse/0/cameras.bin
```

其中：

- `images/00001.jpg` 只保存图像像素。
- `sparse/0/images.bin` 保存每张图片的 COLMAP 外参：`image_id`、`qvec`、`tvec`、`camera_id`、`name`。
- `sparse/0/cameras.bin` 保存相机内参。
- 脚本通过 `images.bin` 中的 `name == "00001.jpg"` 找到该图片的真实相机位姿。

## 脚本位置

```text
scripts/make_pose_render_preview.py
```

该脚本会：

- 读取 `sparse/0/images.bin` 和 `sparse/0/cameras.bin`。
- 根据图片名找到对应 COLMAP camera pose。
- 加载训练好的 `point_cloud.ply`。
- 渲染同位姿图片。
- 固定相机中心，对相机朝向做 `yaw -8°` 和 `yaw +8°` 的轻微偏转，再各渲染一张。
- 拼接原图、同视角渲染、两个偏转视角渲染，并在底部写入相机信息。

## 使用 H100 环境生成 Preview

当前机器上裸跑 PyTorch 时 `cuda:0` 可能映射到 L20。为了确保使用 H100，需要加：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
```

生成 `00001.jpg` 的 preview：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
conda run -n half_gaussian_splatting_h100 \
python scripts/make_pose_render_preview.py \
  -s t2data/Train_colmap \
  -m output/Train_3dhgs_h100 \
  --image 00001.jpg \
  --iteration 30000
```

默认输出：

```text
output/Train_3dhgs_h100/previews/pose_render_preview_00001.jpg
```

## 指定输出路径

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
conda run -n half_gaussian_splatting_h100 \
python scripts/make_pose_render_preview.py \
  -s t2data/Train_colmap \
  -m output/Train_3dhgs_h100 \
  --image 00001.jpg \
  --iteration 30000 \
  --output output/Train_3dhgs_h100/previews/my_preview.jpg
```

## 调整偏转角度

默认偏转角度是 `8` 度。可以用 `--yaw` 修改：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
conda run -n half_gaussian_splatting_h100 \
python scripts/make_pose_render_preview.py \
  -s t2data/Train_colmap \
  -m output/Train_3dhgs_h100 \
  --image 00001.jpg \
  --iteration 30000 \
  --yaw 5
```

## 生成其他图片的 Preview

例如生成 `00151.jpg`：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
conda run -n half_gaussian_splatting_h100 \
python scripts/make_pose_render_preview.py \
  -s t2data/Train_colmap \
  -m output/Train_3dhgs_h100 \
  --image 00151.jpg \
  --iteration 30000
```

输出会是：

```text
output/Train_3dhgs_h100/previews/pose_render_preview_00151.jpg
```

## 注意事项

- 这份 preview 是基于 COLMAP 相机位姿生成的，不是按文件名随便抽样。
- “同视角渲染”使用原图在 `images.bin` 中对应的 `qvec/tvec`。
- “偏转视角渲染”保持相机中心不变，只改变相机朝向。
- 如果 `--image` 指定的图片不在 `images.bin` 中，脚本会报错，因为无法确定它的真实相机位姿。
- 如果模型目录下没有对应的 `point_cloud/iteration_*/point_cloud.ply`，脚本也会报错。
