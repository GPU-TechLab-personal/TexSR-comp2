#!/usr/bin/env python3
"""Export what PBR-SR itself feeds to its SR init, for our models to super-resolve.

    cd docker && docker compose run --rm pbrsr python /workspace/variants/pbr_sr/init/export_lr.py

The LR maps are not re-derived here. The official loader builds them, so they are
exactly the run's LR input:
  load_mesh_data(data_dir, downsample_scale=4) is the call run_pbr_sr.py:708 makes.
  It treats the shipped 4096^2 textures as HR GT and makes the 1024^2 LR maps
  (pbr_sr/utils_online.py:402-477).

The AO channel is also the official one. The study keeps the README's ks init for
AO, i.e. HiT-SRF x4 on the LR ks:
  - the model is built as run_pbr_sr.py:184-188 builds it;
  - it is applied as run_pbr_sr.py:1366 applies it, net_g(ks.permute(0,3,1,2)).clamp(0,1);
  - channel 0 is used, as run_pbr_sr.py:1376 uses it.
For Cgtrader the loader stores ks[...,0] = 1 - AO (utils_online.py:367). The
`read` init loads PNGs without flipping, so that channel is written as it is.

Written to runs/pbr_sr/init/lr/ (--out):
  lr16/{baseColor,roughness,metallic,normal,height}.png  16-bit, MatUp's input
  lr8/{baseColor,roughness,metallic,normal}.png         8-bit, DeepTextures' input
      (its decoder, tf.io.decode_image(channels=3), is the one training used, on
      8-bit PNGs)
  ao_hit4.png    8-bit 4096^2: HiT-SRF x4 of LR ks[...,0] (= 1 - AO, as `read` wants)
  mask.png       8-bit 4096^2: the loader's HR UV mask (albedo alpha > 0)
  meta.json      shapes, value ranges, sources

The value domains are the loader's own:
  - kd: the PNG values, no sRGB<->linear conversion (texture.load_texture2D).
  - roughness / metallic: ks[...,1] / ks[...,2].
  - normal: stored as (n + 1) / 2, because `read` decodes x * 2 - 1
    (run_pbr_sr.py:1727).
  - height: Cradle_Globe ships none, so it is a constant 0.5. MatUp displaces by
    disp_val * (h - 0.5), so this means "no displacement".
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3))
PBRSR = os.path.join(REPO, "external", "PBR-SR")
DATA = "./data/LR_mesh/Cgtrader/Cradle_Globe"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA, help="relative to external/PBR-SR (read only)")
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "pbr_sr", "init", "lr"))
    ap.add_argument("--scale", type=int, default=4, help="--pbr_sr_scale of the runs")
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    os.chdir(PBRSR)                      # the official code resolves ./weights, ./configs
    sys.path.insert(0, PBRSR)
    import numpy as np
    import torch
    from PIL import Image
    from pbr_sr.utils_online import load_mesh_data
    from pbr_sr.utils import setup_basicsr_namespace, yaml_load
    from pbr_sr.paths import PBR_SR_CONFIG_DIR

    device = "cuda"
    mesh_gt, mesh_lr, hr_mask = load_mesh_data(args.data_dir, downsample_scale=args.scale, device=device)
    kd = mesh_lr.material.kd.data[0].float()       # [H,W,3]
    ks = mesh_lr.material.ks.data[0].float()       # [H,W,3] = (1-AO, rough, metal)
    n = mesh_lr.material.normal.data[0].float()    # [H,W,3] in [-1,1]
    H, W = kd.shape[:2]

    def save(path, x, bits):
        x = x.clamp(0, 1).cpu().numpy()
        if bits == 16:
            arr = np.rint(x * 65535).astype(np.uint16)
        else:
            arr = np.rint(x * 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if bits == 16 and arr.ndim == 3:
            # PIL cannot write 16-bit RGB; cv2 is in the image (BGR order).
            import cv2
            cv2.imwrite(path, arr[..., ::-1])
        else:
            Image.fromarray(arr).save(path)

    maps = {
        "baseColor": kd,
        "roughness": ks[..., 1:2],
        "metallic": ks[..., 2:3],
        "normal": (n + 1) / 2,
    }
    for name, x in maps.items():
        save(os.path.join(out, "lr16", name + ".png"), x, 16)
        save(os.path.join(out, "lr8", name + ".png"), x, 8)
    save(os.path.join(out, "lr16", "height.png"), torch.full_like(ks[..., :1], 0.5), 16)

    # HiT-SRF x4 on the LR ks, exactly as the official CAMixer_HiT_SR_x4 init does.
    setup_basicsr_namespace(["./third_party/HiT_SR", "./third_party/CAMixerSR/codes"])
    from basicsr.models import build_model
    opt = yaml_load(os.path.join(PBR_SR_CONFIG_DIR, "hitsr", "test_single_x4.yml"))
    opt["is_train"] = False
    opt["dist"] = False
    hit = build_model(opt)
    with torch.no_grad():
        hit.net_g.eval()
        hit4 = hit.net_g(mesh_lr.material.ks.data.permute(0, 3, 1, 2)).clamp(0.0, 1.0)
    ao = hit4[0, 0][..., None]                     # [4H,4W,1] = 1 - AO, as `read` wants
    save(os.path.join(out, "ao_hit4.png"), ao, 8)
    save(os.path.join(out, "mask.png"), hr_mask[0].float(), 8)

    meta = {
        "data_dir": args.data_dir,
        "scale": args.scale,
        "lr_size": [H, W],
        "hr_size": list(mesh_gt.material.kd.data.shape[-3:-1]) if mesh_gt.material.kd.data.dim() == 4
                   else list(mesh_gt.material.kd.data.shape[:2]),
        "ao_hit4_size": list(ao.shape[:2]),
        "mask_fraction": float(hr_mask.float().mean()),
        "ranges": {k: [float(v.min()), float(v.max())] for k, v in
                   {"kd": kd, "ks_1-ao": ks[..., 0], "rough": ks[..., 1], "metal": ks[..., 2],
                    "normal": n, "ao_hit4": ao}.items()},
        "sources": {
            "lr": "pbr_sr/utils_online.py load_mesh_data(downsample_scale=%d)" % args.scale,
            "ao_hit4": "HiT-SRF x4 (configs/hitsr/test_single_x4.yml) on LR ks, channel 0",
            "height": "constant 0.5 (none shipped)",
        },
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
