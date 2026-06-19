# 输入图像 Preview 生成方法

本文档记录如何为 `t2data/Train_colmap` 生成输入图像预览，方便确认训练数据到底是什么场景。

## 生成内容

脚本会在输出目录中创建：

```text
output/Train_3dhgs/previews/
  input_contact_sheet.jpg
  input_first_00001.jpg
  input_middle_00151.jpg
  input_last_00301.jpg
```

其中：

- `input_contact_sheet.jpg` 是多张输入图像组成的拼图。
- `input_first_00001.jpg` 是第一张输入图像的缩略预览。
- `input_middle_00151.jpg` 是中间视角图像的缩略预览。
- `input_last_00301.jpg` 是最后一张输入图像的缩略预览。

## 运行脚本

在仓库根目录执行：

```bash
cat > .make_previews.py <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

outdir = Path("output/Train_3dhgs/previews")
outdir.mkdir(parents=True, exist_ok=True)

imgs = sorted(Path("t2data/Train_colmap/images").glob("*.jpg"))
print("num images", len(imgs))

for path in imgs[:5] + imgs[len(imgs) // 2:len(imgs) // 2 + 2] + imgs[-3:]:
    image = Image.open(path)
    print(path, image.size, image.mode)

# 选择若干代表性视角生成拼图。
selected_indices = [0, 8, 16, 40, 80, 120, 160, 200, 240, 280, 300]
selected_paths = [imgs[index] for index in selected_indices if index < len(imgs)]

thumbs = []
for path in selected_paths:
    image = Image.open(path).convert("RGB")
    image.thumbnail((320, 180))

    canvas = Image.new("RGB", (320, 210), "white")
    canvas.paste(image, ((320 - image.width) // 2, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 185), path.name, fill=(0, 0, 0))
    thumbs.append(canvas)

cols = 3
rows = (len(thumbs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 320, rows * 210), (240, 240, 240))

for index, thumb in enumerate(thumbs):
    x = (index % cols) * 320
    y = (index // cols) * 210
    sheet.paste(thumb, (x, y))

sheet.save(outdir / "input_contact_sheet.jpg", quality=95)

# 额外保存首张、中间、末张图像的较大缩略预览。
for label, index in [("first", 0), ("middle", len(imgs) // 2), ("last", len(imgs) - 1)]:
    image = Image.open(imgs[index]).convert("RGB")
    image.thumbnail((960, 540))
    image.save(outdir / f"input_{label}_{imgs[index].stem}.jpg", quality=95)

print("preview dir", outdir)
PY

conda run -n half_gaussian_splatting python .make_previews.py
rm -f .make_previews.py
```

## 查看生成结果

列出生成文件：

```bash
find output/Train_3dhgs/previews -maxdepth 1 -type f -print
```

打开拼图：

```bash
xdg-open output/Train_3dhgs/previews/input_contact_sheet.jpg
```

如果当前环境没有桌面图像查看器，也可以把该文件下载到本地查看。

## 本次 Train 场景结果

本次 `t2data/Train_colmap` 的输入数据为：

- 图片数量：`301`
- 图片格式：`RGB`
- 图片分辨率：`980x545`
- 场景内容：Tanks&Temples 的 `Train` 火车场景，一辆 Western Pacific 713 火车的多视角照片。

生成的拼图路径：

```text
output/Train_3dhgs/previews/input_contact_sheet.jpg
```
