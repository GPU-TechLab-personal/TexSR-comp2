#!/usr/bin/env bash
# Environment setup on the lab PC (Windows + Docker Desktop, run in Git Bash; Linux works too),
# for everything or for one part. Every step is idempotent: if anything fails or the connection
# drops, run the same command again and it continues where it stopped.
#
#   bash scripts/setup.sh [PART ...] 2>&1 | tee -a runs/logs/setup.log
#
#   PART                what it sets up                                   needed by
#   data                dataprep image (CPU only)                         scripts/data.sh, init.sh pack
#   deeptex_renloss_21  deeptex image (TF 2.10) + a GPU op. The first     scripts/deeptex_renloss_21.sh,
#                       start JIT-compiles TF for sm_120 (~7-10 min,      init.sh deeptex_renloss_21
#                       once; cached in runs/cache/nv)
#   matup               MatUp's SwinIR-L x4 GAN weight + matup image      scripts/matup_ft.sh,
#                       (torch 2.7.1) + a GPU op                          init.sh matup / matup_ft
#   pbrsr               DiffBIR (3 files, ~6.7 GB), CAMixerSR, HiT-SRF    PBR-SR runs, init.sh export
#                       weights, pbrsr image (~29 GB), HiT-SRF .pth       (CAMixerSR / HiT-SRF only for
#                       conversion, PBR-SR environment gate               the readme preset's ks init)
#   (none) = all four, in the order above.
#
# Every PART first runs the common part: GPU + driver >= 570, docker, compose, docker/.env,
# MatSynth file count, and the submodules at their pinned commits (recursively).
# With several PARTs, a failed one does not stop the others; the summary at the end lists
# each PART as OK or FAILED, and the exit code is non-zero if any failed.
set -uo pipefail
export MSYS_NO_PATHCONV=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p runs/logs runs/cache/nv
PBRSR=external/PBR-SR
ALL="data deeptex_renloss_21 matup pbrsr"
step() { echo; echo "=== [$(date +%T)] $*"; }
die()  { echo "SETUP FAILED: $*" >&2; exit 1; }
if docker compose version >/dev/null 2>&1; then DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then DC="docker-compose"
else DC=""; fi

# fetch URL DEST SIZE  (resumable, exact size)
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

# ------------------------------------------------------------------ common
common() {
  step "common: preflight"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found (no NVIDIA driver)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  local drv; drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "$drv" -ge 570 ] || die "driver $drv < 570: CUDA 12.8 needs >= 570"
  command -v docker >/dev/null || die "docker not installed"
  docker info >/dev/null 2>&1 || die "docker is not running (start Docker Desktop) or needs sudo"
  [ -n "$DC" ] || die "neither 'docker compose' nor 'docker-compose' is available"
  echo "compose: $($DC version | head -1)"
  [ -f docker/.env ] || die "docker/.env missing: cp docker/.env.example docker/.env and set DAS_ROOT"
  local das n; das=$(sed -n 's/^DAS_ROOT=//p' docker/.env | tail -1)
  n=$(ls "$das"/MatSynth/data/*.parquet 2>/dev/null | wc -l)
  echo "DAS_ROOT=$das   MatSynth parquet files: $n (431 train + 8 test when complete)"

  step "common: submodules"
  git submodule update --init --recursive || die "submodule update failed (rerun to resume)"
  git submodule status
}

# ------------------------------------------------------------------ data
part_data() {
  step "data: dataprep image"
  ( cd docker && $DC build dataprep ) || die "dataprep image build failed (rerun: the layer cache resumes it)"
  ( cd docker && $DC run --rm -T dataprep python -c "import pyarrow, cv2, numpy; print('dataprep OK | pyarrow', pyarrow.__version__, '| cv2', cv2.__version__)" ) \
    || die "dataprep image does not start"
}

# ------------------------------------------------------------------ deeptex_renloss_21
part_deeptex_renloss_21() {
  step "deeptex_renloss_21: deeptex image"
  ( cd docker && $DC build deeptex ) || die "deeptex image build failed (rerun: the layer cache resumes it)"
  step "deeptex_renloss_21: GPU (first start JIT-compiles TF for this GPU - several minutes, once)"
  ( cd docker && $DC run --rm -T deeptex python /workspace/scripts/tf_gpu_test.py ) || die "deeptex image cannot use the GPU"
}

# ------------------------------------------------------------------ matup
part_matup() {
  step "matup: weights"
  [ -f external/matup_egsr/sr_modules/SwinIR/main_test_swinir.py ] || die "MatUp's SwinIR submodule missing"
  # SwinIR-L x4 GAN (external/matup_egsr/pretrained_models/download_links.txt)
  fetch "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth" \
        external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth 142473939
  step "matup: image"
  ( cd docker && $DC build matup ) || die "matup image build failed (rerun: the layer cache resumes it)"
  step "matup: GPU"
  ( cd docker && $DC run --rm -T matup python -c "
import torch; x = torch.randn(4, 3, 64, 64, device='cuda'); c = torch.nn.Conv2d(3, 8, 3).cuda()
y = c(x); y.mean().backward(); print('matup GPU OK |', torch.cuda.get_device_name(), torch.cuda.get_arch_list())" ) \
    || die "matup image cannot use the GPU"
}

# ------------------------------------------------------------------ pbrsr
part_pbrsr() {
  step "pbrsr: memory"
  # Docker Desktop's VM memory is what the containers get, not the host RAM. The PBR-SR
  # pseudo-GT step keeps all 750 views in RAM (~33 GB and more while it writes the
  # cache): a 30 GB limit was OOM-killed at view 652 (2026-11-15).
  local vm_gb; vm_gb=$(docker info --format '{{.MemTotal}}' 2>/dev/null | awk '{printf "%d", $1/1073741824}')
  echo "docker VM memory: ${vm_gb} GB"
  [ "${vm_gb:-0}" -ge 48 ] || echo "WARN: docker has under 48 GB - raise it (COMMANDS.txt: .wslconfig) or PBR-SR is OOM-killed"
  [ -f "$PBRSR/data/LR_mesh/Cgtrader/Cradle_Globe/mesh.obj" ] || die "PBR-SR example data missing"

  step "pbrsr: weights"
  # DiffBIR: the files DiffBIR downloads at first run
  # (third_party/DiffBIR/diffbir/inference/pretrained_models.py); sizes as downloaded.
  local HF=https://huggingface.co/lxq007/DiffBIR-v2/resolve/main D=$PBRSR/third_party/DiffBIR/weights
  fetch "$HF/realesrgan_s4_swinir_100k.pth"      "$D/realesrgan_s4_swinir_100k.pth"      90208287
  fetch "$HF/DiffBIR_v2.1.pt"                    "$D/DiffBIR_v2.1.pt"                    1452744042
  fetch "$HF/sd2.1-base-zsnr-laionaes5.ckpt"     "$D/sd2.1-base-zsnr-laionaes5.ckpt"     5160249169
  # CAMixerSR: the official guide copies them out of the submodule (weights/README.md).
  mkdir -p "$PBRSR/weights/CAMixerSR"
  cp -n "$PBRSR"/third_party/CAMixerSR/pretrained_models/LightSR/*.pth "$PBRSR/weights/CAMixerSR/"
  # HiT-SRF: the HF repos the official guide names (converted to .pth below).
  local s H
  for s in 2 4; do
    H=runs/pbr_sr/cache/hit_sr_hf/hit-srf-${s}x
    fetch "https://huggingface.co/XiangZ/hit-srf-${s}x/resolve/main/config.json" "$H/config.json" 579
  done
  fetch "https://huggingface.co/XiangZ/hit-srf-2x/resolve/main/model.safetensors" runs/pbr_sr/cache/hit_sr_hf/hit-srf-2x/model.safetensors 3495728
  fetch "https://huggingface.co/XiangZ/hit-srf-4x/resolve/main/model.safetensors" runs/pbr_sr/cache/hit_sr_hf/hit-srf-4x/model.safetensors 3573632

  step "pbrsr: image"
  ( cd docker && $DC build pbrsr ) || die "pbrsr image build failed (rerun: the layer cache resumes it)"
  ( cd docker && $DC run --rm -T pbrsr nvidia-smi -L ) || die "pbrsr container does not see the GPU"

  step "pbrsr: HiT-SRF conversion + environment gate"
  if [ -f "$PBRSR/weights/HiT_SR/HiT-SRF-4x.pth" ] && [ -f "$PBRSR/weights/HiT_SR/HiT-SRF-2x.pth" ]; then
    echo "HiT-SRF already converted"
  else
    ( cd docker && $DC run --rm -T pbrsr python /workspace/variants/pbr_sr/prep_hitsr_weights.py ) \
      || die "prep_hitsr_weights.py failed"
  fi
  ( cd docker && $DC run --rm -T pbrsr python /workspace/variants/pbr_sr/gates/gate_env.py --part image ) \
    || die "PBR-SR environment gate failed (runs/pbr_sr/gate/env_image.json)"
}

# ------------------------------------------------------------------ main
PARTS=${*:-$ALL}
for p in $PARTS; do
  case " $ALL " in *" $p "*) ;; *) sed -n '2,23p' "$0"; echo "unknown PART: $p" >&2; exit 1 ;; esac
done
( common ) || exit 1
summary=() rc=0
for p in $PARTS; do
  if ( "part_$p" ); then summary+=("OK      $p"); else summary+=("FAILED  $p"); rc=1; fi
done
step "summary"
printf '%s\n' "${summary[@]}"
[ $rc = 0 ] && echo "SETUP DONE: $PARTS" || echo "SETUP INCOMPLETE - rerun the FAILED part(s): bash scripts/setup.sh <PART>"
exit $rc
