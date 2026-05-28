# OpenSora ViDiT-Q 环境、权重与 T5 预计算说明

本文档记录当前服务器上的推荐流程：

1. 从头安装 OpenSora + ViDiT-Q 依赖。
2. 在本地下载模型权重，并上传到服务器。
3. 使用已补充的 T5 text embedding 预计算逻辑，避免量化/推理阶段反复加载 T5。

服务器代码目录：

```bash
/root/data-fs/viditq
```

服务器 conda 环境建议放在：

```bash
/root/data-fs/conda_envs/opensora_viditq
```

模型权重统一放在：

```bash
/root/data-fs/models
```

---

## 1. 从头安装依赖

### 1.1 创建 conda 环境到 data-fs

服务器系统盘空间有限，不要把环境装到默认 `/root/miniconda3/envs`。

```bash
mkdir -p /root/data-fs/conda_envs
mkdir -p /root/data-fs/conda_pkgs
mkdir -p /root/data-fs/cache/pip
mkdir -p /root/data-fs/cache/tmp
mkdir -p /root/data-fs/cache/huggingface
mkdir -p /root/data-fs/cache/torch
mkdir -p /root/data-fs/models
```

```bash
conda config --add envs_dirs /root/data-fs/conda_envs
conda config --add pkgs_dirs /root/data-fs/conda_pkgs
```

```bash
conda create -y -p /root/data-fs/conda_envs/opensora_viditq python=3.10
source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/data-fs/conda_envs/opensora_viditq && source /root/data-fs/codex_env.sh
```

### 1.2 配置缓存目录

当前 shell 先设置：

```bash
export TMPDIR=/root/data-fs/cache/tmp
export PIP_CACHE_DIR=/root/data-fs/cache/pip
export HF_HOME=/root/data-fs/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/data-fs/cache/huggingface/hub
export TRANSFORMERS_CACHE=/root/data-fs/cache/huggingface/transformers
export DIFFUSERS_CACHE=/root/data-fs/cache/huggingface/diffusers
export TORCH_HOME=/root/data-fs/cache/torch
export XDG_CACHE_HOME=/root/data-fs/cache
```

写入 conda 环境激活脚本：

```bash
mkdir -p /root/data-fs/conda_envs/opensora_viditq/etc/conda/activate.d

cat > /root/data-fs/conda_envs/opensora_viditq/etc/conda/activate.d/cache_paths.sh <<'EOF'
export TMPDIR=/root/data-fs/cache/tmp
export PIP_CACHE_DIR=/root/data-fs/cache/pip
export HF_HOME=/root/data-fs/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/data-fs/cache/huggingface/hub
export TRANSFORMERS_CACHE=/root/data-fs/cache/huggingface/transformers
export DIFFUSERS_CACHE=/root/data-fs/cache/huggingface/diffusers
export TORCH_HOME=/root/data-fs/cache/torch
export XDG_CACHE_HOME=/root/data-fs/cache
EOF
```

### 1.3 安装基础工具

```bash
python -m pip install -U pip setuptools wheel packaging ninja
```

### 1.4 安装 OpenSora 依赖

```bash
cd /root/data-fs/viditq/examples/opensora1.2/Open-Sora
```

安装 CUDA 12.1 对应的 torch / torchvision / xformers：

```bash
python -m pip install --no-cache-dir -r requirements/requirements-cu121.txt
```

安装 OpenSora 依赖：

```bash
python -m pip install --no-cache-dir -r requirements/requirements.txt
```

说明：

- `requirements.txt` 里有 `gradio>=4.26.0`，pip 可能安装到很新的 gradio，并要求 `huggingface-hub>=0.33.5`。
- 本项目量化流程不需要 gradio。
- OpenSora / diffusers / transformers 这套环境更适合保留 `huggingface-hub==0.21.4`。

因此安装完依赖后，固定 HF hub 版本：

```bash
python -m pip install --no-cache-dir "huggingface-hub==0.21.4"
```

安装 OpenSora 本体时跳过依赖解析，避免 gradio 的依赖冲突阻塞安装：

```bash
python -m pip install --no-cache-dir --no-deps -v -e .
```

### 1.5 安装 ViDiT-Q 量化包

```bash
cd /root/data-fs/viditq/quant_utils
python -m pip install --no-cache-dir -e .
```

注意：`quant_utils/setup.py` 已修正为 `find_namespace_packages(include=["qdiff*"])`，否则原始代码因缺少 `__init__.py` 会导致 `find_packages()` 找不到 `qdiff`。

### 1.6 不安装 ViDiT-Q CUDA kernel

当前服务器如果使用 V100，不能安装/使用：

```bash
/root/data-fs/viditq/kernels
```

原因是 `kernels/setup.py` 只支持 `8.0/8.6/8.7/8.9/9.0`，V100 是 `sm70`。

因此不要执行：

```bash
cd /root/data-fs/viditq/kernels
python -m pip install -e .
```

### 1.7 检查关键包

```bash
python - <<'PY'
mods = [
    "torch",
    "torchvision",
    "xformers",
    "colossalai",
    "mmengine",
    "timm",
    "omegaconf",
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "opensora",
    "qdiff",
]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "FAIL", type(e).__name__, e)
PY
```

---

## 2. 本地下载权重并上传到服务器

服务器连不上 HuggingFace，因此不要在服务器上执行 `hf download`。

需要下载 4 个模型：

```text
hpcai-tech/OpenSora-STDiT-v3
hpcai-tech/OpenSora-VAE-v1.2
PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers
DeepFloyd/t5-v1_1-xxl
```

其中：

- STDiT：量化核心模型，必须有。
- OpenSora-VAE：时间 VAE，latent shape 和视频解码必须有。
- PixArt VAE：OpenSora-VAE 依赖的 2D spatial VAE，缺少时会尝试联网访问 HuggingFace 的 `PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/vae/config.json`。
- T5：服务器上运行 `precompute_text_embeds.py` 时必须有。

### 2.1 本地安装 HF CLI

在本地机器执行：

```bash
pip install -U huggingface_hub
```

如果需要登录：

```bash
hf auth login
```

### 2.2 本地下载模型

在本地机器执行：

```bash
cd ~/vscode

unset HF_ENDPOINT
unset HF_HUB_DISABLE_XET

mkdir -p ./models/hpcai-tech/OpenSora-STDiT-v3
mkdir -p ./models/hpcai-tech/OpenSora-VAE-v1.2
mkdir -p ./models/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers
mkdir -p ./models/DeepFloyd/t5-v1_1-xxl
```

下载 STDiT：

```bash
hf download hpcai-tech/OpenSora-STDiT-v3 \
  --local-dir ./models/hpcai-tech/OpenSora-STDiT-v3
```

下载 VAE：

```bash
hf download hpcai-tech/OpenSora-VAE-v1.2 \
  --local-dir ./models/hpcai-tech/OpenSora-VAE-v1.2
```

下载 PixArt spatial VAE：

```bash
hf download PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers \
  --include "vae/*" \
  --local-dir ./models/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers
```

下载 T5：

```bash
hf download DeepFloyd/t5-v1_1-xxl \
  --local-dir ./models/DeepFloyd/t5-v1_1-xxl
```

下载完成后，本地目录应为：

```text
~/vscode/models/hpcai-tech/OpenSora-STDiT-v3
~/vscode/models/hpcai-tech/OpenSora-VAE-v1.2
~/vscode/models/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers
~/vscode/models/DeepFloyd/t5-v1_1-xxl
```

### 2.3 上传到服务器

你的服务器登录方式：

```bash
ssh -p 42102 root@10.137.144.40
```

本地执行：

```bash
ssh -p 42102 root@10.137.144.40 "mkdir -p /root/data-fs/models"
```

```bash
rsync -avP -e "ssh -p 42102" ./models/ root@10.137.144.40:/root/data-fs/models/
```

服务器最终路径应为：

```text
/root/data-fs/models/hpcai-tech/OpenSora-STDiT-v3
/root/data-fs/models/hpcai-tech/OpenSora-VAE-v1.2
/root/data-fs/models/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers
/root/data-fs/models/DeepFloyd/t5-v1_1-xxl
```

### 2.4 服务器配置路径

已修改：

```text
/root/data-fs/viditq/examples/opensora1.2/configs/software_simulation.py
/root/data-fs/viditq/examples/opensora1.2/configs/cuda_kernel.py
```

其中模型路径为：

```python
model = dict(
    from_pretrained="/root/data-fs/models/hpcai-tech/OpenSora-STDiT-v3",
)

vae = dict(
    from_pretrained="/root/data-fs/models/hpcai-tech/OpenSora-VAE-v1.2",
    vae_2d_from_pretrained="/root/data-fs/models/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    local_files_only=True,
)

text_encoder = dict(
    from_pretrained="/root/data-fs/models/DeepFloyd/t5-v1_1-xxl",
)

model_path = "/root/data-fs/models"
```

---

## 3. T5 text embedding 预计算逻辑与使用方法

### 3.1 修改过的文件

新增：

```text
/root/data-fs/viditq/examples/opensora1.2/precompute_text_embeds.py
/root/data-fs/viditq/examples/opensora1.2/text_embed_utils.py
```

修改：

```text
/root/data-fs/viditq/examples/opensora1.2/Open-Sora/opensora/schedulers/rf/__init__.py
/root/data-fs/viditq/examples/opensora1.2/Open-Sora/opensora/utils/config_utils.py
/root/data-fs/viditq/examples/opensora1.2/fp_inference.py
/root/data-fs/viditq/examples/opensora1.2/get_calib_data.py
/root/data-fs/viditq/examples/opensora1.2/quant_inference.py
/root/data-fs/viditq/examples/opensora1.2/configs/software_simulation.py
```

### 3.2 设计逻辑

原始代码里 `precompute_text_embeds=True` 时，只会读固定文件：

```text
./precomputed_text_embeds.pth
```

并且不校验 prompt 是否匹配。换 prompt 集合时很容易错误复用旧 embedding。

现在改为：

1. 先用 `precompute_text_embeds.py` 对指定 prompt 文件预计算 T5 embedding。
2. 输出一个 `.pth` 文件，例如：

```text
./text_embeds/t2v_samples.pth
```

3. 推理/校准时传入：

```bash
--precompute-text-embeds True
--precomputed-text-embeds-path ./text_embeds/t2v_samples.pth
```

4. scheduler 会按当前 batch 的 prompt 字符串，从 `.pth` 里的 `text_embeds` 映射中取对应 embedding。
5. 如果当前 prompt 在 `.pth` 中不存在，会直接报错，避免静默用错 embedding。
6. unconditional/null embedding 不保存在 `.pth` 中，而是在模型所在 GPU 上通过 `model.y_embedder` 生成，再与 T5 conditional embedding 拼接。

这样做的好处：

- T5 只在预计算阶段加载。
- `get_calib_data.py`、`ptq.py`、`quant_inference.py` 阶段不需要加载 T5。
- 不同 prompt 文件对应不同 `.pth`，不会混用。

### 3.3 预计算 text embeddings

假设 prompt 文件为：

```text
/root/data-fs/viditq/examples/opensora1.2/assets/t2v_samples.txt
```

在服务器上执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/data-fs/conda_envs/opensora_viditq && source /root/data-fs/codex_env.sh
cd /root/data-fs/viditq/examples/opensora1.2
mkdir -p text_embeds
```

如果有多卡，建议用 `device_map` 把 T5 分摊到多张卡。当前服务器是 3 张 16GB V100，推荐：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python precompute_text_embeds.py \
  configs/software_simulation.py \
  --prompt-path ./assets/t2v_samples.txt \
  --output ./text_embeds/t2v_samples.pth \
  --batch-size 1 \
  --device cuda:0 \
  --device-map auto \
  --max-memory-per-gpu 14GiB \
  --offload-folder ./text_embeds/t5_offload
```

说明：只写 `CUDA_VISIBLE_DEVICES=0,1` 不会自动分摊显存；原始脚本会把完整 T5 放到 `cuda:0`。`--device-map auto` 才会让 Transformers/Accelerate 按层切分到可见 GPU。

如果只有单卡 16GB，T5-XXL 可能仍然不够；可以尝试 CPU/disk offload，但速度会明显变慢：

```bash
python precompute_text_embeds.py \
  configs/software_simulation.py \
  --prompt-path ./assets/t2v_samples.txt \
  --output ./text_embeds/t2v_samples.pth \
  --batch-size 1 \
  --device cuda:0 \
  --device-map auto \
  --max-memory-per-gpu 14GiB \
  --offload-folder ./text_embeds/t5_offload
```

换 prompt 文件时必须重新预计算，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python precompute_text_embeds.py \
  configs/software_simulation.py \
  --prompt-path ./assets/my_prompts.txt \
  --output ./text_embeds/my_prompts.pth \
  --batch-size 1 \
  --device cuda:0 \
  --device-map auto \
  --max-memory-per-gpu 14GiB \
  --offload-folder ./text_embeds/t5_offload
```

### 3.4 使用预计算 embeddings 跑校准、PTQ、量化推理

用 cuda2 跑模型，避免 T5 占用模型卡：

```bash
CUDA_VISIBLE_DEVICES=2 python get_calib_data.py configs/software_simulation.py \
  --prompt-path ./assets/t2v_samples.txt \
  --precompute-text-embeds True \
  --precomputed-text-embeds-path ./text_embeds/t2v_samples.pth
```

```bash
CUDA_VISIBLE_DEVICES=2 python ptq.py configs/software_simulation.py \
  --save-dir ./logs/w4a8_mp \
  --precompute-text-embeds True \
  --precomputed-text-embeds-path ./text_embeds/t2v_samples.pth
```

```bash
CUDA_VISIBLE_DEVICES=2 python quant_inference.py configs/software_simulation.py \
  --save-dir ./logs/w4a8_mp \
  --precompute-text-embeds True \
  --precomputed-text-embeds-path ./text_embeds/t2v_samples.pth
```

说明：

- `get_calib_data.py` 的 prompt 集合决定校准 activation 分布。
- 如果换了校准 prompt，必须重新跑 `precompute_text_embeds.py` 和 `get_calib_data.py`。
- 如果只换最终生成 prompt，不改校准 prompt，则只需要给新 prompt 文件重新生成对应 text embedding，并在 `quant_inference.py` 时使用新的 `.pth`。

### 3.5 配置里的默认字段

`configs/software_simulation.py` 已增加：

```python
precompute_text_embeds = False
precomputed_text_embeds_path = "./text_embeds/t2v_samples.pth"
```

也可以不改配置，直接用命令行覆盖：

```bash
--precompute-text-embeds True
--precomputed-text-embeds-path ./text_embeds/t2v_samples.pth
```

### 3.6 V100 注意事项

V100 不建议使用 flash-attn，`software_simulation.py` 中已关闭：

```python
enable_flash_attn=False
```

V100 也不支持本仓库的 ViDiT-Q CUDA kernel，因此使用 `software_simulation.py` 跑软件仿真量化。

---

## 4. 按 README video 流程直接执行

本节只给固定执行流程。使用当前代码时，README 的 video 流程前面需要先执行一次 T5 text embedding 预计算。之后 FP16 推理、calibration、PTQ、量化推理都读取这份 `.pth`，不再加载完整 T5。

本流程固定使用：

```text
prompt 文件: ./assets/t2v_samples.txt
text embedding: ./text_embeds/t2v_samples.pth
PTQ/量化目录: ./logs/w4a8_mp
```

直接按顺序执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/data-fs/conda_envs/opensora_viditq && source /root/data-fs/codex_env.sh
cd /root/data-fs/viditq/examples/opensora1.2
mkdir -p text_embeds logs
```

确认权重已经在服务器上：

```bash
ls /root/data-fs/models/hpcai-tech/OpenSora-STDiT-v3
ls /root/data-fs/models/hpcai-tech/OpenSora-VAE-v1.2
ls /root/data-fs/models/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/vae
ls /root/data-fs/models/DeepFloyd/t5-v1_1-xxl
```

### 4.1 预计算 T5 text embeddings

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python precompute_text_embeds.py \
  configs/software_simulation.py \
  --prompt-path ./assets/t2v_samples.txt \
  --output ./text_embeds/t2v_samples.pth \
  --batch-size 1 \
  --device cuda:0 \
  --device-map auto \
  --max-memory-per-gpu 14GiB \
  --offload-folder ./text_embeds/t5_offload
```

### 4.2 FP16 推理

```bash
CUDA_VISIBLE_DEVICES=2 python fp_inference.py configs/software_simulation.py \
  --prompt-path ./assets/t2v_samples.txt \
  --save-dir ./logs/fp16 \
  --precompute-text-embeds True \
  --precomputed-text-embeds-path ./text_embeds/t2v_samples.pth
```

### 4.3 生成 calibration data

```bash
CUDA_VISIBLE_DEVICES=2 python get_calib_data.py configs/software_simulation.py \
  --prompt-path ./assets/t2v_samples.txt \
  --save-dir ./logs/get_calib_data \
  --precompute-text-embeds True \
  --precomputed-text-embeds-path ./text_embeds/t2v_samples.pth
```

这一步会按照 `configs/w4a8_mixed_precision.yaml` 里的配置生成：

```text
./calib_data.pth
```

### 4.4 PTQ

```bash
CUDA_VISIBLE_DEVICES=2 python ptq.py configs/software_simulation.py \
  --save-dir ./logs/w4a8_mp \
  --precompute-text-embeds True \
  --precomputed-text-embeds-path ./text_embeds/t2v_samples.pth
```

这一步会生成：

```text
./logs/w4a8_mp/quant_params.pth
```

### 4.5 量化推理

```bash
CUDA_VISIBLE_DEVICES=2 python quant_inference.py configs/software_simulation.py \
  --prompt-path ./assets/t2v_samples.txt \
  --save-dir ./logs/w4a8_mp \
  --precompute-text-embeds True \
  --precomputed-text-embeds-path ./text_embeds/t2v_samples.pth
```

### 4.6 输出位置

FP16 推理输出：

```text
./logs/fp16
```

calibration data：

```text
./calib_data.pth
```

PTQ 参数和量化推理输出：

```text
./logs/w4a8_mp
```

---

## 5. 复现 Table 1 的 VBench 三个指标

本节用于只评测 OpenSora/VBench 中 Table 1 需要的三个维度：

```text
subject_consistency
overall_consistency
scene
```

校准仍然复用前面 `./assets/t2v_samples.txt` 的 10 条 prompt 生成的结果，不需要为了 VBench prompt 重新跑 calibration。

已从 `Open-Sora/eval/vbench/VBench_full_info.json` 抽取出 prompt 文件：

```text
./assets/vbench/subject_consistency.txt    72 prompts
./assets/vbench/overall_consistency.txt    93 prompts
./assets/vbench/scene.txt                  86 prompts
./assets/vbench/table1_selected.txt        251 prompts
```

### 5.1 确认已有 calibration/PTQ 结果

必须已经有：

```bash
cd /root/data-fs/viditq/examples/opensora1.2
ls ./text_embeds/t2v_samples.pth
ls ./calib_data.pth
ls ./logs/w4a8_mp/quant_params.pth
```

其中 `quant_params.pth` 是基于原来的 10 条 calibration prompt 得到的。后续 VBench 推理只复用它。

### 5.2 新建 VBench 评测 conda 环境

推理、PTQ、量化推理仍然使用：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/data-fs/conda_envs/opensora_viditq && source /root/data-fs/codex_env.sh
```

VBench 算分单独使用新环境，避免污染 OpenSora/ViDiT-Q 环境：

```bash
mkdir -p /root/data-fs/conda_envs
mkdir -p /root/data-fs/conda_pkgs
mkdir -p /root/data-fs/cache/pip
mkdir -p /root/data-fs/cache/tmp
mkdir -p /root/data-fs/models/vbench

conda config --add envs_dirs /root/data-fs/conda_envs
conda config --add pkgs_dirs /root/data-fs/conda_pkgs

conda create -y -p /root/data-fs/conda_envs/vbench_eval python=3.10
source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/data-fs/conda_envs/vbench_eval && source /root/data-fs/codex_env.sh

export TMPDIR=/root/data-fs/cache/tmp
export PIP_CACHE_DIR=/root/data-fs/cache/pip
export VBENCH_CACHE_DIR=/root/data-fs/models/vbench
```

安装 PyTorch 和 VBench 三个指标需要的依赖：

```bash
python -m pip install -U pip setuptools wheel packaging

python -m pip install --no-cache-dir \
  torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install --no-cache-dir \
  decord opencv-python-headless pillow tqdm numpy scipy scikit-image \
  timm fairscale einops ftfy regex transformers==4.36.2 huggingface-hub==0.21.4
```

OpenAI CLIP 建议从本地上传源码后安装，见 5.3。上传后在服务器执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/data-fs/conda_envs/vbench_eval && source /root/data-fs/codex_env.sh
python -m pip install --no-cache-dir /root/data-fs/src/CLIP
```

VBench 代码已经在本仓库：

```text
/root/data-fs/viditq/eval/video/Vbench
```

评测时通过环境变量使用本地 VBench 代码和本地评测权重：

```bash
export PYTHONPATH=/root/data-fs/viditq/eval/video/Vbench:$PYTHONPATH
export VBENCH_CACHE_DIR=/root/data-fs/models/vbench
```

### 5.3 本地下载并上传 VBench 评测权重

这三个指标会用到：

```text
subject_consistency: DINO
overall_consistency: ViCLIP
scene: Tag2Text
```

服务器最终应有这些文件：

```text
/root/data-fs/models/vbench/dino_model/facebookresearch_dino_main/
/root/data-fs/models/vbench/dino_model/dino_vitbase16_pretrain.pth
/root/data-fs/models/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth
/root/data-fs/models/vbench/ViCLIP/bpe_simple_vocab_16e6.txt.gz
/root/data-fs/models/vbench/caption_model/tag2text_swin_14m.pth
/root/data-fs/src/CLIP
```

在本地机器执行：

```bash
cd ~/vscode

python -m pip install -U huggingface_hub hf_xet

mkdir -p ./vbench_ckpts/dino_model
mkdir -p ./vbench_ckpts/ViCLIP
mkdir -p ./vbench_ckpts/caption_model
mkdir -p ./src

git clone https://github.com/facebookresearch/dino \
  ./vbench_ckpts/dino_model/facebookresearch_dino_main

wget -O ./vbench_ckpts/dino_model/dino_vitbase16_pretrain.pth \
  https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth

hf download OpenGVLab/VBench_Used_Models \
  ViClip-InternVid-10M-FLT.pth \
  --local-dir ./vbench_ckpts/ViCLIP

wget -O ./vbench_ckpts/ViCLIP/bpe_simple_vocab_16e6.txt.gz \
  https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz

wget -O ./vbench_ckpts/caption_model/tag2text_swin_14m.pth \
  https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth

git clone https://github.com/openai/CLIP ./src/CLIP
```

上传到服务器：

```bash
ssh -p 42102 root@10.137.144.40 "mkdir -p /root/data-fs/models/vbench /root/data-fs/src"

rsync -avP -e "ssh -p 42102" \
  ./vbench_ckpts/ \
  root@10.137.144.40:/root/data-fs/models/vbench/

rsync -avP -e "ssh -p 42102" \
  ./src/CLIP \
  root@10.137.144.40:/root/data-fs/src/
```

服务器上确认：

```bash
ls /root/data-fs/models/vbench/dino_model/facebookresearch_dino_main
ls /root/data-fs/models/vbench/dino_model/dino_vitbase16_pretrain.pth
ls /root/data-fs/models/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth
ls /root/data-fs/models/vbench/ViCLIP/bpe_simple_vocab_16e6.txt.gz
ls /root/data-fs/models/vbench/caption_model/tag2text_swin_14m.pth
ls /root/data-fs/src/CLIP
```

### 5.4 为 VBench prompt 预计算 T5 embeddings

VBench prompt 和 calibration prompt 不同，所以必须给 `table1_selected.txt` 重新生成 text embedding：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/data-fs/conda_envs/opensora_viditq && source /root/data-fs/codex_env.sh
cd /root/data-fs/viditq/examples/opensora1.2
mkdir -p text_embeds

CUDA_VISIBLE_DEVICES=0 python precompute_text_embeds.py \
  configs/software_simulation.py \
  --prompt-path ./assets/vbench/table1_selected.txt \
  --output ./text_embeds/vbench_table1_selected.pth \
  --batch-size 1 \
  --device cuda:0 \
  --device-map auto \
  --max-memory-per-gpu 14GiB \
  --offload-folder ./text_embeds/t5_offload
```

如果当前机器有多张可用 GPU，可以把 `CUDA_VISIBLE_DEVICES=0` 改为 `CUDA_VISIBLE_DEVICES=0,1,2`。

### 5.5 用旧 calibration 结果跑量化推理

`quant_inference.py` 会从 `save_dir/quant_params.pth` 读取量化参数。为了把 VBench 输出单独放目录里，同时复用旧 PTQ 结果，先建立软链接：

```bash
cd /root/data-fs/viditq/examples/opensora1.2
mkdir -p ./logs/w4a8_mp/vbench_table1
ln -sf ../quant_params.pth ./logs/w4a8_mp/vbench_table1/quant_params.pth
```

然后生成 VBench 视频。这里必须使用 `--prompt-as-path`，因为 VBench 按 `prompt.mp4` 文件名匹配视频：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python quant_inference.py configs/software_simulation.py \
  --prompt-path ./assets/vbench/table1_selected.txt \
  --save-dir ./logs/w4a8_mp/vbench_table1 \
  --prompt-as-path \
  --precompute-text-embeds True \
  --precomputed-text-embeds-path ./text_embeds/vbench_table1_selected.pth
```

输出视频目录：

```text
/root/data-fs/viditq/examples/opensora1.2/logs/w4a8_mp/vbench_table1
```

### 5.6 计算三个 VBench 指标

从 Open-Sora 目录运行评测脚本：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/data-fs/conda_envs/vbench_eval && source /root/data-fs/codex_env.sh
cd /root/data-fs/viditq/examples/opensora1.2/Open-Sora

export PYTHONPATH=/root/data-fs/viditq/eval/video/Vbench:$PYTHONPATH
export VBENCH_CACHE_DIR=/root/data-fs/models/vbench

CUDA_VISIBLE_DEVICES=0 python eval/vbench/calc_vbench.py \
  /root/data-fs/viditq/examples/opensora1.2/logs/w4a8_mp/vbench_table1 \
  /root/data-fs/viditq/examples/opensora1.2/logs/w4a8_mp/vbench_table1_scores \
  --dimensions subject_consistency overall_consistency scene \
  --local
```

结果会写到：

```text
/root/data-fs/viditq/examples/opensora1.2/logs/w4a8_mp/vbench_table1_scores/vbench
```

包括：

```text
subject_consistency_eval_results.json
overall_consistency_eval_results.json
scene_eval_results.json
```

只评测这三个维度时，不要直接跑原始 `tabulate_vbench_scores.py`，因为它要求 16 个 VBench 维度都存在。读取三个 raw score：

```bash
python - <<'PY'
import json
from pathlib import Path
score_dir = Path("/root/data-fs/viditq/examples/opensora1.2/logs/w4a8_mp/vbench_table1_scores/vbench")
for dim in ["subject_consistency", "overall_consistency", "scene"]:
    path = score_dir / f"{dim}_eval_results.json"
    data = json.load(path.open())
    print(dim, data[dim][0])
PY
```

### 5.7 如果也要跑 FP16 baseline

同样复用 VBench text embeddings，但不需要量化参数：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python fp_inference.py configs/software_simulation.py \
  --prompt-path ./assets/vbench/table1_selected.txt \
  --save-dir ./logs/fp16/vbench_table1 \
  --prompt-as-path \
  --precompute-text-embeds True \
  --precomputed-text-embeds-path ./text_embeds/vbench_table1_selected.pth
```

然后把 `calc_vbench.py` 的 video folder 和 score folder 分别换成：

```text
/root/data-fs/viditq/examples/opensora1.2/logs/fp16/vbench_table1
/root/data-fs/viditq/examples/opensora1.2/logs/fp16/vbench_table1_scores
```
