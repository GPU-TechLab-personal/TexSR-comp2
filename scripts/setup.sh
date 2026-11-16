#!/usr/bin/env bash
# One-time setup on the lab PC (Windows + Docker Desktop, run in Git Bash; Linux works too).
# Every step is idempotent: if anything fails or the connection drops, run it again and
# it continues where it stopped.
#
#   bash scripts/setup.sh 2>&1 | tee runs/logs/setup.log
#
#   1 preflight  GPU + driver >= 570, docker (no sudo), compose, RAM, free disk, docker/.env
#   2 code       submodules at their pinned commits, recursively (MatUp's SwinIR, PBR-SR's
#                third_party/*)
#   3 weights    DiffBIR (3 files), CAMixerSR (copied from its submodule), HiT-SRF (HF),
#                SwinIR-L x4 GAN for MatUp - each from the URL its own README gives,
#                resumable and size-checked
#   4 images     docker compose build: dataprep, deeptex, matup, pbrsr
#   5 GPU        one real GPU op in each GPU image. deeptex: the first start JIT-compiles
#                TF's kernels for sm_120 (~7-10 min, once; cached in runs/cache/nv)
#   6 PBR-SR     HiT-SRF .pth conversion + official loader check, environment gate
set -uo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p runs/logs runs/cache/nv
PBRSR=external/PBR-SR
step() { echo; echo "=== [$(date +%T)] $*"; }
die()  { echo "SETUP FAILED: $*" >&2; exit 1; }
if docker compose version >/dev/null 2>&1; then DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then DC="docker-compose"
else DC=""; fi

# fetch NAME_FOR_LOG URL DEST SIZE  (resumable, exact size)
fetch() {
  local url=$1 dst=$2 size=$3
  if [ -f "$dst" ] && [ "$(wc -c < "$dst" | tr -d ' ')" = "$size" ]; then echo "ok   $dst"; return 0; fi
  echo "get  $dst"
  mkdir -p "$(dirname "$dst")"
  for try in 1 2 3 4 5; do
    curl -fL --retry 5 --retry-delay 10 -C - -o "$dst.partial" "$url" && break
    echo "  retry $try"; sleep 10
  done
  [ "$(wc -c < "$dst.partial" | tr -d ' ')" = "$size" ] || die "$dst: size mismatch (rerun to resume)"
  mv "$dst.partial" "$dst"
}

# ------------------------------------------------------------------ 1 preflight
step "1 preflight"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (no NVIDIA driver)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$drv" -ge 570 ] || die "driver $drv < 570: CUDA 12.8 needs >= 570"
command -v docker >/dev/null || die "docker not installed"
docker info >/dev/null 2>&1 || die "docker is not running (start Docker Desktop) or needs sudo"
[ -n "$DC" ] || die "neither 'docker compose' nor 'docker-compose' is available"
echo "compose: $($DC version | head -1)"
# Docker Desktop's VM memory is what the containers get, not the host RAM. The
# PBR-SR pseudo-GT step keeps all 750 views in RAM (~33 GB and more while it writes
# the cache): a 30 GB limit was OOM-killed at view 652 (2026-11-15).
vm_gb=$(docker info --format '{{.MemTotal}}' 2>/dev/null | awk '{printf "%d", $1/1073741824}')
echo "docker VM memory: ${vm_gb} GB"
[ "${vm_gb:-0}" -ge 48 ] || echo "WARN: docker has under 48 GB - raise it (COMMANDS.txt: .wslconfig) or PBR-SR is OOM-killed"
[ -f docker/.env ] || die "docker/.env missing: cp docker/.env.example docker/.env and set DAS_ROOT"
das=$(sed -n 's/^DAS_ROOT=//p' docker/.env | tail -1)
echo "DAS_ROOT=$das"
n=$(ls "$das"/MatSynth/data/*.parquet 2>/dev/null | wc -l)
echo "MatSynth parquet files under $das/MatSynth/data: $n (431 train + 8 test when complete)"

# ------------------------------------------------------------------ 2 code
step "2 code (submodules)"
git submodule update --init --recursive || die "submodule update failed (rerun to resume)"
git submodule status
[ -f "$PBRSR/data/LR_mesh/Cgtrader/Cradle_Globe/mesh.obj" ] || die "PBR-SR example data missing"
[ -f external/matup_egsr/sr_modules/SwinIR/main_test_swinir.py ] || die "MatUp's SwinIR submodule missing"

# ------------------------------------------------------------------ 3 weights
step "3 weights"
# DiffBIR: the files DiffBIR downloads at first run
# (third_party/DiffBIR/diffbir/inference/pretrained_models.py); sizes as downloaded.
HF=https://huggingface.co/lxq007/DiffBIR-v2/resolve/main
D=$PBRSR/third_party/DiffBIR/weights
fetch "$HF/realesrgan_s4_swinir_100k.pth"      "$D/realesrgan_s4_swinir_100k.pth"      90208287
fetch "$HF/DiffBIR_v2.1.pt"                    "$D/DiffBIR_v2.1.pt"                    1452744042
fetch "$HF/sd2.1-base-zsnr-laionaes5.ckpt"     "$D/sd2.1-base-zsnr-laionaes5.ckpt"     5160249169
# CAMixerSR: the official guide copies them out of the submodule (weights/README.md).
mkdir -p "$PBRSR/weights/CAMixerSR"
cp -n "$PBRSR"/third_party/CAMixerSR/pretrained_models/LightSR/*.pth "$PBRSR/weights/CAMixerSR/"
# HiT-SRF: the HF repos the official guide names (converted to .pth in step 6).
for s in 2 4; do
  H=runs/pbr_sr/cache/hit_sr_hf/hit-srf-${s}x
  fetch "https://huggingface.co/XiangZ/hit-srf-${s}x/resolve/main/config.json" "$H/config.json" 579
done
fetch "https://huggingface.co/XiangZ/hit-srf-2x/resolve/main/model.safetensors" runs/pbr_sr/cache/hit_sr_hf/hit-srf-2x/model.safetensors 3495728
fetch "https://huggingface.co/XiangZ/hit-srf-4x/resolve/main/model.safetensors" runs/pbr_sr/cache/hit_sr_hf/hit-srf-4x/model.safetensors 3573632
# MatUp: SwinIR-L x4 GAN (external/matup_egsr/pretrained_models/download_links.txt).
fetch "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth" \
      external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth 142473939

# ------------------------------------------------------------------ 4 images
step "4 images"
( cd docker && $DC build dataprep deeptex matup pbrsr ) || die "image build failed (rerun: the layer cache resumes it)"

# ------------------------------------------------------------------ 5 GPU
step "5 GPU in each image"
( cd docker && $DC run --rm -T matup python -c "
import torch; x = torch.randn(4, 3, 64, 64, device='cuda'); c = torch.nn.Conv2d(3, 8, 3).cuda()
y = c(x); y.mean().backward(); print('matup GPU OK |', torch.cuda.get_device_name(), torch.cuda.get_arch_list())" ) \
  || die "matup image cannot use the GPU"
echo "deeptex: first start JIT-compiles TF for this GPU - several minutes, once"
( cd docker && $DC run --rm -T deeptex python /workspace/scripts/tf_gpu_test.py ) || die "deeptex image cannot use the GPU"
( cd docker && $DC run --rm -T pbrsr nvidia-smi -L ) || die "pbrsr container does not see the GPU"

# ------------------------------------------------------------------ 6 PBR-SR
step "6 PBR-SR"
if [ -f "$PBRSR/weights/HiT_SR/HiT-SRF-4x.pth" ] && [ -f "$PBRSR/weights/HiT_SR/HiT-SRF-2x.pth" ]; then
  echo "HiT-SRF already converted"
else
  ( cd docker && $DC run --rm -T pbrsr python /workspace/variants/pbr_sr/prep_hitsr_weights.py ) \
    || die "prep_hitsr_weights.py failed"
fi
( cd docker && $DC run --rm -T pbrsr python /workspace/variants/pbr_sr/gates/gate_env.py --part image ) \
  || die "PBR-SR environment gate failed (runs/pbr_sr/gate/env_image.json)"

step "SETUP DONE"
