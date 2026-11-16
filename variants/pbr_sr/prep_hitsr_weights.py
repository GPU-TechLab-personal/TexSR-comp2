#!/usr/bin/env python3
"""Put the HiT-SRF weights where the official PBR-SR configs expect them, and prove
the official loader accepts them.

The official guide (external/PBR-SR/weights/README.md) says to download the model
file from https://huggingface.co/XiangZ/hit-srf-{2,4}x and "save or rename it" to
weights/HiT_SR/HiT-SRF-{2,4}x.pth. That repo serves only model.safetensors, while the
official loader (third_party/HiT_SR/basicsr/models/base_model.py:290, via
configs/hitsr/test_single_x{2,4}.yml) does torch.load(path)["params"]. This script
  1. records what a renamed safetensors file does under torch.load (the literal
     reading of the guide),
  2. writes {"params": <the safetensors state dict>} to the expected .pth path - the
     same tensors, only the container format changes,
  3. builds the model exactly as run_pbr_sr.py:113-116, 184-188 does (strict_load_g:
     true), and runs one forward pass.

Run in the pbrsr service (downloads are already in runs/pbr_sr/cache/hit_sr_hf/):
    docker compose run --rm pbrsr python /workspace/variants/pbr_sr/prep_hitsr_weights.py
"""
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
PBRSR = os.path.join(REPO, "external", "PBR-SR")
RAW = os.path.join(REPO, "runs", "pbr_sr", "cache", "hit_sr_hf")


def main():
    import torch
    from safetensors.torch import load_file

    os.chdir(PBRSR)                      # the official code resolves ./weights, ./configs
    sys.path.insert(0, PBRSR)
    from pbr_sr.utils import setup_basicsr_namespace, yaml_load
    from pbr_sr.paths import PBR_SR_CONFIG_DIR
    setup_basicsr_namespace(["./third_party/HiT_SR", "./third_party/CAMixerSR/codes"])
    from basicsr.models import build_model

    report = {}
    os.makedirs(os.path.join("weights", "HiT_SR"), exist_ok=True)
    for s in ("2", "4"):
        src = os.path.join(RAW, "hit-srf-%sx" % s, "model.safetensors")
        dst = os.path.join("weights", "HiT_SR", "HiT-SRF-%sx.pth" % s)
        r = {"source": os.path.relpath(src, REPO)}

        with tempfile.TemporaryDirectory() as tmp:
            renamed = os.path.join(tmp, "HiT-SRF-%sx.pth" % s)
            shutil.copy(src, renamed)
            try:
                obj = torch.load(renamed, map_location="cpu")
                r["renamed_torch_load"] = "loaded: %s" % type(obj).__name__
            except Exception as e:  # noqa: BLE001 - recording what the literal guide does
                r["renamed_torch_load"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])

        sd = load_file(src)
        torch.save({"params": sd}, dst)
        r["written"] = dst
        r["tensors"] = len(sd)

        opt = yaml_load(os.path.join(PBR_SR_CONFIG_DIR, "hitsr", "test_single_x%s.yml" % s))
        opt["is_train"] = False
        opt["dist"] = False
        model = build_model(opt)        # strict_load_g: true in the config
        net = model.net_g.eval()
        with torch.no_grad():
            out = net(torch.rand(1, 3, 32, 32, device=next(net.parameters()).device))
        r["forward_out_shape"] = list(out.shape)
        r["ok"] = r["forward_out_shape"] == [1, 3, 32 * int(s), 32 * int(s)]
        report["x" + s] = r

    path = os.path.join(REPO, "runs", "pbr_sr", "gate", "hitsr_weights.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(0 if all(r["ok"] for r in report.values()) else 1)


if __name__ == "__main__":
    main()
