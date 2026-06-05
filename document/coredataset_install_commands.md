# coredataset 环境安装命令

本文档给出 `coredataset` 环境的推荐安装命令。建议一段一段执行，执行完每段都看一下是否报错。

## 1. 推荐版本

推荐组合：

- Python: `3.12`
- PyTorch: `2.10.0`
- torchvision: `0.25.0`
- torchaudio: `2.10.0`
- CUDA wheel: `cu128`
- ffmpeg: `7.1.1`
- LeRobot: `0.5.1`

依据：

- LeRobot 官方安装文档要求 Python `>=3.12`，并要求支持 PyTorch `>=2.10`。
- PyTorch 官方安装页提供 `torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0` 的 CUDA 12.8 wheel。
- 本机 `nvidia-smi` 提权检查显示驱动可用，适合优先尝试 CUDA 12.8 wheel。

参考链接：

- https://huggingface.co/docs/lerobot/installation
- https://pypi.org/project/lerobot/
- https://pytorch.org/get-started/previous-versions/

## 2. 网络与代理策略

当前推荐策略：默认直连，优先使用国内镜像；只有访问 GitHub、Hugging Face、arXiv 等外网失败时，再在当前终端手动开启代理。

如果需要代理，Docker 使用 `--network host` 时，容器内应使用服务器侧实际监听端口 `127.0.0.1:7890`。Windows 本机梯子端口可能是 `7897`，但容器内不一定能直接访问 `127.0.0.1:7897`。

默认取消代理：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY=localhost,127.0.0.1,::1
```

需要访问 GitHub / Hugging Face 时再开启代理：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY=localhost,127.0.0.1,::1
```

检查网络是否可用：

```bash
curl -I --max-time 10 https://repo.anaconda.com/pkgs/main/noarch/repodata.json
curl -I --max-time 10 https://download.pytorch.org/whl/cu128/
curl -I --max-time 10 https://huggingface.co/
```

如果你想用项目脚本加载代理，也可以执行：

```bash
JIAFENG_USE_PROXY=1 source /home/xiaobo.xia/JiafengWu/env.sh
```

但如果它把代理设到不可用端口，请重新执行本节开头那组 `export`，覆盖成 `7890`。

## 3. 安装容器系统依赖

`lerobot[aloha]` 会通过 `pynput` 依赖 `evdev`。在 `continuumio/miniconda3:latest` 的 Debian 容器里，`evdev` 可能需要本地编译；如果缺少 Linux input 头文件，会报：

```text
The 'linux/input.h' and 'linux/input-event-codes.h' include files are missing.
```

因此进入 Docker 后，先以 root 在容器内安装系统编译依赖：

```bash
apt-get update
apt-get install -y --no-install-recommends build-essential linux-libc-dev
```

如果启动容器时使用了 `--user $(id -u):$(id -g)`，当前 shell 不是 root，可以在宿主机另开终端执行：

```bash
docker exec -u root <container-id-or-name> bash -lc 'apt-get update && apt-get install -y --no-install-recommends build-essential linux-libc-dev'
```

说明：如果使用 `--rm`，这些 apt 安装只保存在当前容器实例里。退出容器后重开，需要重新执行本节，或者自己构建一个包含这些系统依赖的 Docker 镜像。

## 4. 创建或进入 coredataset 环境

先检查环境是否已经存在：

```bash
conda env list | grep coredataset
```

如果已经存在，直接激活：

```bash
conda activate coredataset
```

如果不存在，创建环境：

```bash
conda create -y -n coredataset -c conda-forge python=3.12 pip ffmpeg=7.1.1
conda activate coredataset
```

检查基础版本：

```bash
python --version
ffmpeg -version | head -n 1
python -m pip --version
```

## 5. 安装 PyTorch CUDA 12.8

先升级基础安装工具：

```bash
python -m pip install -U pip setuptools wheel
```

安装 PyTorch。国内网络下优先使用南京大学 PyTorch 镜像，并用清华 PyPI 镜像补普通依赖：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
python -m pip install "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128" \
  --index-url https://mirror.nju.edu.cn/pytorch/whl/cu128/ \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

如果镜像不可用，再退回 PyTorch 官方源：

```bash
python -m pip install "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

验证 PyTorch 和 CUDA：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

## 6. 安装项目上层依赖

安装 LeRobot 和核心实验包。这里建议让 `pip` 一次性解析所有上层依赖，不要把 `numpy`、`opencv`、`lerobot` 分成互相独立的强 pin 安装，否则容易出现局部版本满足、整体版本冲突。

```bash
python -m pip install \
  datasets==4.0.0 \
  huggingface-hub==1.3.0 \
  transformers==5.0.0 \
  tokenizers==0.22.1 \
  safetensors==0.7.0 \
  accelerate==1.12.0 \
  open-clip-torch==3.2.0 \
  scikit-learn==1.8.0 \
  pandas==2.3.3 \
  matplotlib==3.10.8 \
  seaborn==0.13.2 \
  tqdm==4.67.1 \
  rich==14.2.0 \
  numpy==2.2.6 \
  scipy==1.16.3 \
  pillow==12.0.0 \
  opencv-python==4.12.0.88 \
  pyarrow==22.0.0 \
  h5py==3.15.1 \
  "lerobot[aloha]==0.5.1" \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

说明：

- `lerobot[aloha]` 用于 ALOHA 数据集与相关工具。
- `open-clip-torch` 用于 CLIP 特征提取。
- `scikit-learn` 用于 MLP baseline、聚类、k-center / coverage 等传统实验组件。
- `transformers==5.0.0` 要求 `huggingface-hub>=1.3.0,<2.0`，因此这里使用 `huggingface-hub==1.3.0`。不要改回 `1.2.3`。
- `opencv-python==4.12.0.88` 要求 `numpy>=2,<2.3.0`，`lerobot==0.5.1` 也要求 `numpy>=2.0.0,<2.3.0`，因此这里使用 `numpy==2.2.6`。不要改回 `2.3.5`。
- 如果想先检查依赖能否解析，可以把上面的命令临时加上 `--dry-run`。
- 如果后续 `pip` 报依赖冲突，以 `pip` 输出为准，不要强行 `--force-reinstall`，先把报错保存下来。

## 7. 验证关键包

```bash
python - <<'PY'
import torch
import torchvision
import datasets
import transformers
import sklearn
import pandas
import matplotlib
import open_clip

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)
print("sklearn:", sklearn.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)
print("open_clip imported")
print("cuda available:", torch.cuda.is_available())
PY
```

验证 LeRobot：

```bash
python - <<'PY'
import lerobot
print("lerobot imported:", lerobot.__file__)
PY
```

## 8. Hugging Face 数据集连通性检查

```bash
python - <<'PY'
from huggingface_hub import HfApi

repo_id = "lerobot/aloha_sim_transfer_cube_human"
info = HfApi().dataset_info(repo_id)
print("dataset:", info.id)
print("sha:", info.sha)
print("siblings:", len(info.siblings))
PY
```

如果这里失败，多半是 Hugging Face 网络或代理问题，先重新检查第 2 节代理。

## 9. 导出环境记录

安装成功后，建议导出两份记录，后面写报告或复现实验都能用：

```bash
conda env export -n coredataset > document/coredataset_env_full.yml
python -m pip freeze > document/coredataset_pip_freeze.txt
```

## 10. 如果 PyTorch CUDA 12.8 安装失败

优先不要混装多个 CUDA 版本。先卸载 PyTorch 三件套：

```bash
python -m pip uninstall -y torch torchvision torchaudio
```

然后尝试 CUDA 12.6 wheel：

```bash
python -m pip install "torch==2.10.0+cu126" "torchvision==0.25.0+cu126" "torchaudio==2.10.0+cu126" \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

再运行第 5 节的 CUDA 验证命令。
