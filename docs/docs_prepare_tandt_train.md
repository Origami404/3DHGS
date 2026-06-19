# Tanks&Temples Train 数据准备记录

本文档记录如何为本仓库准备论文复现用的单个 Tanks&Temples 场景：`Train`。

## 目标格式

本仓库训练入口 `train.py` 需要 3D-GS/COLMAP 风格的数据目录。对于单个场景，目录应类似：

```text
t2data/Train_colmap/
  images/
    00001.jpg
    00002.jpg
    ...
  sparse/
    0/
      cameras.bin
      images.bin
      points3D.bin
      points3D.ply
```

其中：

- `images/` 存放该场景的所有图像。
- `sparse/0/cameras.bin` 存放 COLMAP 相机内参。
- `sparse/0/images.bin` 存放 COLMAP 相机外参和图片名。
- `sparse/0/points3D.bin` 存放 COLMAP 稀疏点云。
- `sparse/0/points3D.ply` 可由本仓库首次读取 `points3D.bin` 时自动生成。

## 下载数据

使用 3D Gaussian Splatting 官方提供的预处理数据包 `T&T+DB COLMAP`。该包已经包含 Tanks&Temples 的 `train` 和 `truck` 场景，以及 Deep Blending 的场景，因此不需要重新跑 COLMAP。

```bash
mkdir -p t2data/downloads

python - <<'PY'
import pathlib
import time
import urllib.request

url = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
out = pathlib.Path("t2data/downloads/tandt_db.zip")

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as response, out.open("wb") as handle:
    total = int(response.headers.get("Content-Length") or 0)
    downloaded = 0
    last_report = 0
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        handle.write(chunk)
        downloaded += len(chunk)
        now = time.time()
        if now - last_report > 2:
            print(f"下载 {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB", flush=True)
            last_report = now

print(f"完成 {downloaded} bytes")
PY
```

下载完成后，可以检查文件是否为有效 zip：

```bash
python - <<'PY'
import zipfile
from pathlib import Path

path = Path("t2data/downloads/tandt_db.zip")
print("bytes:", path.stat().st_size)
print("is_zip:", zipfile.is_zipfile(path))

with zipfile.ZipFile(path) as archive:
    roots = sorted({"/".join(name.strip("/").split("/")[:2]) for name in archive.namelist() if len(name.strip("/").split("/")) >= 2})
    print("scenes:")
    for root in roots:
        print("-", root)
PY
```

正常情况下会看到：

```text
db/drjohnson
db/playroom
tandt/train
tandt/truck
```

## 解压单个场景

这里只抽取论文复现可用的 Tanks&Temples `Train` 场景，输出到 `t2data/Train_colmap`：

```bash
rm -rf t2data/Train_colmap.tmp
mkdir -p t2data/Train_colmap.tmp

python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

archive_path = Path("t2data/downloads/tandt_db.zip")
output_dir = Path("t2data/Train_colmap.tmp")
prefix = "tandt/train/"

count = 0
with ZipFile(archive_path) as archive:
    for member in archive.infolist():
        name = member.filename
        if not name.startswith(prefix) or name == prefix:
            continue

        relative = name[len(prefix):]
        if not relative:
            continue

        target = output_dir / relative
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            count += 1

print("extracted files", count)
PY

rm -rf t2data/Train_colmap
mv t2data/Train_colmap.tmp t2data/Train_colmap
```

检查目录：

```bash
find t2data/Train_colmap -maxdepth 2 -type d | sort
find t2data/Train_colmap/images -maxdepth 1 -type f | wc -l
ls -lh t2data/Train_colmap/sparse/0
```

本次准备得到的结果是：

- 图片数量：`301`
- COLMAP 稀疏模型目录：`t2data/Train_colmap/sparse/0`
- 初始点云文件：`t2data/Train_colmap/sparse/0/points3D.bin`

## 读取验证

使用仓库的 conda 环境验证数据能被 `scene.dataset_readers` 正确读取：

```bash
cat > .check_3dhgs_dataset.py <<'PY'
from scene.dataset_readers import readColmapSceneInfo

scene = readColmapSceneInfo("t2data/Train_colmap", images=None, eval=True)
print("train cameras", len(scene.train_cameras))
print("test cameras", len(scene.test_cameras))
print("radius", scene.nerf_normalization["radius"])
print("ply", scene.ply_path)
print("pcd points", None if scene.point_cloud is None else scene.point_cloud.points.shape)
print("first train image", scene.train_cameras[0].image_path, scene.train_cameras[0].width, scene.train_cameras[0].height)
PY

conda run -n half_gaussian_splatting python .check_3dhgs_dataset.py
rm -f .check_3dhgs_dataset.py
```

本次验证输出的关键结果：

```text
train cameras 263
test cameras 38
pcd points (182686, 3)
first train image t2data/Train_colmap/images/00002.jpg 1959 1090
```

首次读取时，代码会自动从 `points3D.bin` 生成：

```text
t2data/Train_colmap/sparse/0/points3D.ply
```

## 开始训练

```bash
conda activate half_gaussian_splatting
python train.py -s t2data/Train_colmap --eval -m output/Train_3dhgs
```

说明：

- `-s t2data/Train_colmap` 指向准备好的 COLMAP 场景目录。
- `--eval` 按代码默认逻辑每 8 张图取 1 张作为测试集。
- `-m output/Train_3dhgs` 指定模型输出目录。

## 测试和评分

训练完成后运行：

```bash
python test_and_score.py -s t2data/Train_colmap -m output/Train_3dhgs
```

该命令会渲染测试视角，并输出 PSNR、SSIM、LPIPS 等指标。

## 注意事项

- `t2data/image_sets/*.zip` 中通过旧 Google Drive 脚本下载到的文件可能是登录 HTML，而不是真正 zip。可以用 `zipfile.is_zipfile(...)` 检查。
- 如果文件开头类似 `<!doctype html><html ... accounts.google.com ...>`，说明下载失败，需要丢弃。
- 当前文档使用的 `tandt_db.zip` 是已预处理的 COLMAP 版本，不需要再运行 `convert.py`。
- 如果要改用另一个论文场景 `Truck`，把解压前缀从 `tandt/train/` 改成 `tandt/truck/`，输出目录可改为 `t2data/Truck_colmap`。
