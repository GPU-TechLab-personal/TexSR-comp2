#!/usr/bin/env python3
"""Gate G0 env: the official PBR-SR code (external/PBR-SR, never edited) runs on the
pbrsr service (texsr/pbrsr, docker/pbrsr: torch 2.7.1 / CUDA 12.8, RTX 5080).

Two parts, as in variants/sr_finetune/gates/gate_env.py:

  --part host    on the host, where git sees the checkout as it really is:
                     python variants/pbr_sr/gates/gate_env.py --part host
                 -> runs/pbr_sr/gate/env_host.json
  --part image   in the pbrsr service:
                     docker compose run --rm pbrsr python /workspace/variants/pbr_sr/gates/gate_env.py --part image
                 -> runs/pbr_sr/gate/env_image.json

host
  pin         external/PBR-SR HEAD is PIN
  clean       `git status --porcelain` is empty in external/PBR-SR and its
              third_party/ submodules (run it again after the image part: the
              official code writes caches next to its inputs, never into itself)

image
  versions    torch 2.7.1 with CUDA 12.8, a GPU of compute capability 12.0 visible,
              numpy 1.x
  rasterize   nvdiffrast RasterizeCudaContext rasterizes one triangle (this is also
              where its CUDA plugin is JIT-compiled, into TORCH_EXTENSIONS_DIR)
  renderutils the official nvdiff_render CUDA plugin (JIT-compiled the same way):
              prepare_shading_normal in CUDA equals its PyTorch reference
  attention   DiffBIR resolves to torch SDPA (ATTN_MODE=sdp) and SDPA matches an fp32
              reference; xformers' default dispatch is probed in a child process and
              only recorded (on sm_120 it aborts inside a FlashAttention-3 kernel)
  imports     the official tests/smoke_imports.py, run unchanged from the
              submodule root (imports run_pbr_sr and the DiffBIR backend)

Weights are NOT needed and not touched. Exit 0 on pass, 1 on fail.
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3))
PBRSR = os.path.join(REPO, "external", "PBR-SR")
PIN = "f2faf0f1d9f85113e0c598cfdbcd68c5a24ebf97"
GATE_DIR = os.path.join(REPO, "runs", "pbr_sr", "gate")


def write_json(name, obj):
    os.makedirs(GATE_DIR, exist_ok=True)
    path = os.path.join(GATE_DIR, name)
    with open(path + ".partial", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".partial", path)
    return path


def git(root, *args):
    try:
        return subprocess.check_output(["git", "-C", root] + list(args),
                                       stderr=subprocess.STDOUT).decode()
    except (OSError, subprocess.CalledProcessError) as e:
        return "ERROR " + (e.output.decode() if hasattr(e, "output") and e.output else str(e))


def part_host():
    head = (git(PBRSR, "rev-parse", "HEAD") or "").strip()
    checks = {"pin": {"ok": head == PIN, "head": head, "pinned": PIN}}
    out = git(PBRSR, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none")
    subs = git(PBRSR, "submodule", "foreach", "--quiet", "git status --porcelain --untracked-files=all")
    checks["clean"] = {"ok": out == "" and subs == "", "status": out, "submodules_status": subs}
    return checks


def guarded(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - the point is to report it
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}


def check_versions():
    import numpy as np
    import torch
    v = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "numpy": np.__version__,
         "cuda": torch.cuda.is_available()}
    if v["cuda"]:
        v["gpu"] = torch.cuda.get_device_name(0)
        v["capability"] = list(torch.cuda.get_device_capability(0))
        v["arch_list"] = torch.cuda.get_arch_list()
    v["ok"] = (v["torch"].startswith("2.7.1") and v["cuda_build"] == "12.8" and v["cuda"]
               and v.get("capability") == [12, 0] and v["numpy"].startswith("1."))
    return v


def check_rasterize():
    import torch
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()
    pos = torch.tensor([[[-0.8, -0.8, 0.0, 1.0], [0.8, -0.8, 0.0, 1.0], [0.0, 0.8, 0.0, 1.0]]],
                       device="cuda")
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    rast, _ = dr.rasterize(glctx, pos, tri, resolution=[64, 64])
    covered = int((rast[..., 3] > 0).sum())
    # The triangle covers 0.8 * 1.6 / 2 of the [-1,1]^2 viewport = 32% of 4096 px.
    return {"ok": 1100 < covered < 1500, "covered_px": covered, "nvdiffrast": dr.__name__}


def check_renderutils():
    import torch
    sys.path.insert(0, PBRSR)
    from nvdiff_render.renderutils import ops
    torch.manual_seed(0)
    shape = (1, 32, 32, 3)
    n = lambda t: torch.nn.functional.normalize(t, dim=-1)  # noqa: E731
    pos = torch.randn(shape, device="cuda")
    view = torch.tensor([0.0, 0.0, 5.0], device="cuda")[None, None, None]
    pert = n(torch.randn(shape, device="cuda") * 0.3 + torch.tensor([0, 0, 1.0], device="cuda"))
    nrm = n(torch.randn(shape, device="cuda"))
    tng = n(torch.randn(shape, device="cuda"))
    geo = n(torch.randn(shape, device="cuda"))
    a = ops.prepare_shading_normal(pos, view, pert, nrm, tng, geo, True, True, use_python=False)
    b = ops.prepare_shading_normal(pos, view, pert, nrm, tng, geo, True, True, use_python=True)
    err = float((a - b).abs().max())
    return {"ok": err < 1e-4, "max_abs_diff_cuda_vs_python": err}


XFORMERS_PROBE = r"""
import torch, xformers.ops as xops
q, k, v = (torch.randn(1, 256, 4, 64, device="cuda", dtype=torch.float16) for _ in range(3))
a = xops.memory_efficient_attention(q, k, v)
b = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
print("MAXDIFF", float((a - b).abs().max()))
"""


def check_attention():
    """DiffBIR must run on torch SDPA here (ATTN_MODE=sdp, docker/pbrsr/Dockerfile).

    xformers' default dispatch is probed in a child process and RECORDED, not judged:
    on sm_120 it can abort the process from inside a CUDA kernel, which no
    try/except in this process would catch.
    """
    import torch
    q, k, v = (torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    ref = torch.softmax((q.float() @ k.float().transpose(-1, -2)) / 8.0, -1) @ v.float()
    sdpa_err = float((out.float() - ref).abs().max())

    probe = subprocess.run([sys.executable, "-c", "import os, sys; sys.path.insert(0, %r); "
                            "from diffbir.model.config import Config; print('MODE', Config.attn_mode.name)"
                            % os.path.join(PBRSR, "third_party", "DiffBIR")],
                           capture_output=True, text=True)
    mode = next((l.split()[1] for l in probe.stdout.splitlines() if l.startswith("MODE")), None)

    xf = subprocess.run([sys.executable, "-c", XFORMERS_PROBE], capture_output=True, text=True)
    xf_line = next((l for l in xf.stdout.splitlines() if l.startswith("MAXDIFF")), None)
    return {"ok": sdpa_err < 1e-2 and mode == "SDP", "sdpa_max_abs_diff_vs_fp32": sdpa_err,
            "ATTN_MODE": os.environ.get("ATTN_MODE"), "diffbir_attn_mode": mode,
            "diffbir_probe_stderr_tail": probe.stderr.strip().splitlines()[-3:],
            "xformers_default_dispatch": {"returncode": xf.returncode, "result": xf_line,
                                          "stderr_tail": (xf.stderr.strip().splitlines() or [""])[-1]}}


def check_imports():
    p = subprocess.run([sys.executable, "tests/smoke_imports.py"], cwd=PBRSR,
                       capture_output=True, text=True)
    tail = (p.stdout + p.stderr).strip().splitlines()[-25:]
    return {"ok": p.returncode == 0, "returncode": p.returncode, "tail": tail}


def part_image():
    checks = {"versions": guarded(check_versions)}
    for name, fn in (("rasterize", check_rasterize), ("renderutils", check_renderutils),
                     ("attention", check_attention), ("imports", check_imports)):
        checks[name] = guarded(fn)
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["host", "image"], required=True)
    args = ap.parse_args()
    checks = part_host() if args.part == "host" else part_image()
    report = {"part": args.part, "checks": checks,
              "failed": [k for k, c in checks.items() if not c["ok"]]}
    report["pass"] = not report["failed"]
    path = write_json("env_%s.json" % args.part, report)
    for k, c in checks.items():
        print("  %-12s %s" % (k, "ok" if c["ok"] else "FAIL  " + str(c.get("error", ""))[:200]))
    print("[gate G0 env/%s] %s -> %s" % (args.part, "PASS" if report["pass"] else "FAIL", path))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
