# PBR-SR (Chen et al., NeurIPS 2025) - https://github.com/TerenceCYJ/PBR-SR @ f2faf0f
# (external/PBR-SR, a submodule, never edited).
#
# The official environment is torch 2.2.2 / CUDA 11.8, which has no kernels for the
# RTX 5080 (Blackwell, sm_120). This image keeps the official code and every other
# pin it can, and swaps only the torch stack for one that runs here: torch 2.7.1 /
# CUDA 12.8. Each other difference is listed in requirements.txt with its reason.
#
# The -devel base is on purpose: nvdiffrast and nvdiff_render/renderutils compile
# their CUDA plugins at first import (torch.utils.cpp_extension), so nvcc and a host
# compiler must be in the image. The compiled plugins are cached under
# TORCH_EXTENSIONS_DIR on the /workspace mount (gitignored runs/), so they are built
# once per host, not once per container.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    TORCH_EXTENSIONS_DIR=/workspace/runs/pbr_sr/cache/torch_extensions \
    HF_HOME=/workspace/runs/pbr_sr/cache/hf \
    ATTN_MODE=sdp
# ATTN_MODE: upstream DiffBIR's own switch (third_party/DiffBIR/diffbir/model/config.py:49-60),
# read by both the UNet attention and the VAE (model/vae.py:326). With xformers
# installed DiffBIR defaults to xformers.ops.memory_efficient_attention, which on this
# GPU dispatches to the FlashAttention-3 (Hopper) kernel and aborts the process with
# "CUDA error ... flash_fwd_launch_template.h:188: invalid argument" (measured
# 2026-11-14, xformers 0.0.31.post1, RTX 5080). "sdp" = torch scaled_dot_product_attention,
# the same attention computed by a different kernel.

# git: submodule heads in logs. libgl1 + libglib2.0-0: opencv-python's libGL.so.1.
# build-essential + ninja-build: the JIT CUDA plugins above.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Freeze the base image's torch stack: nothing installed below may replace it.
RUN pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /tmp/torch-constraints.txt \
    && cat /tmp/torch-constraints.txt

COPY requirements.txt /tmp/pbrsr/requirements.txt
RUN pip install -r /tmp/pbrsr/requirements.txt -c /tmp/torch-constraints.txt

# basicsr 1.4.2's setup.py imports torch, so it cannot build in an isolated env.
# Its basicsr/data/degradations.py imports torchvision.transforms.functional_tensor,
# which torchvision removed in 0.17; the official torchvision 0.17.2 fails the same
# way, so this is a fix to the installed dependency, not to PBR-SR. The patched line
# imports the same function from its current module.
RUN pip install --no-build-isolation basicsr==1.4.2 -c /tmp/torch-constraints.txt \
    && f=$(python -c "import importlib.util as u; print(u.find_spec('basicsr').submodule_search_locations[0])")/data/degradations.py \
    && sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$f" \
    && grep -n "rgb_to_grayscale" "$f" | head -2

# nvdiffrast 0.3.3 (the official pin) is not on PyPI; install the same release from
# its tag (v0.3.3 = 729261d). 0.3.x compiles its CUDA plugin at first use, so the
# install itself needs no GPU. --no-build-isolation: setup.py imports torch.
RUN pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8" -c /tmp/torch-constraints.txt

# xformers built for this torch. --no-deps: it must not pull a different torch.
RUN pip install --no-deps xformers==0.0.31.post1 --index-url https://download.pytorch.org/whl/cu128

# Upstream DiffBIR annotates with torch.Tuple (third_party/DiffBIR/diffbir/sampler/
# edm_sampler.py:145, evaluated at class definition). torch 2.2.2 (the official stack)
# re-exported typing.Tuple as torch.Tuple; torch 2.7 does not, so importing DiffBIR
# fails with "module 'torch' has no attribute 'Tuple'". Restore that one name in the
# installed torch (the only torch.<typing name> in external/PBR-SR, grep 2026-11-14),
# rather than edit the submodule.
RUN f=$(python -c "import importlib.util as u; print(u.find_spec('torch').origin)") \
    && printf '\n# texsr/pbrsr: torch 2.2 re-exported typing.Tuple here; upstream DiffBIR uses it.\nfrom typing import Tuple  # noqa: E402,F401\n' >> "$f" \
    && python -c "import torch, typing; assert torch.Tuple is typing.Tuple"

# Fail the build, not the first run: torch is still the base image's, and the
# packages the official code imports at module level resolve.
RUN python -c "import torch, torchvision, xformers, nvdiffrast, basicsr, DISTS_pytorch, torchmetrics, lpips, kornia, pyiqa; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| torchvision', torchvision.__version__, '| xformers', xformers.__version__)" \
    && pip freeze | grep -E '^(torch|torchvision|torchaudio)==' | diff - /tmp/torch-constraints.txt

# The official code resolves weights/, configs/ and third_party/ relative to the
# working directory, so it must run from the submodule root.
WORKDIR /workspace/external/PBR-SR
CMD ["bash"]
