#!/usr/bin/env python3
"""Pack an arm's 4096^2 SR maps into the three files PBR-SR's `read` init loads.

    cd docker && docker compose run --rm dataprep python variants/pbr_sr/init/pack_read.py --arm r1
    cd docker && docker compose run --rm dataprep python variants/pbr_sr/init/pack_read.py --arm matup

What `read` does (run_pbr_sr.py, not edited):
  kd  :1037  texture.load_texture2D(<data_dir>/*sr_kd_output.png)          as is
  ks  :1428  texture.load_texture2D(<data_dir>/*sr_ks_output.png)          as is
  n   :1724  texture.load_texture2D(<data_dir>/*sr_n_output.png, x*2-1)
  load_texture2D -> util.load_image: imageio.imread(...) / 255 (nvdiff_render/util.py:442).
  A 16-bit PNG would therefore come in 257x too bright: the files are 8-bit.

Channels:
  kd  RGB  = the arm's baseColor (the PNG domain, as the loader keeps kd)
  ks  R    = runs/pbr_sr/init/lr/ao_hit4.png: the official HiT-SRF x4 AO, already
             1 - AO as the Cgtrader loader stores it (utils_online.py:367).
             `read` does not flip it.
      G    = the arm's roughness
      B    = the arm's metallic
  n   RGB  = the arm's normal, stored as (n+1)/2 (export_lr.py wrote it that way)

The arms' maps:
  r1     runs/pbr_sr/init/r1/*.png, 8-bit 3-channel (DeepTextures writes RGB even
         for a grey input). Grey maps use the mean of the three channels.
  matup     runs/pbr_sr/init/matup/*.png, 16-bit, blended by matup_tiles.py.
  matup_ft  runs/pbr_sr/init/matup_ft/*.png, the same with the fine-tuned SwinIR-L.

Written to runs/pbr_sr/init/read_<arm>/: sr_kd_output.png, sr_ks_output.png,
sr_n_output.png, meta.json.
"""
import argparse
import json
import os

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3))
INIT = os.path.join(REPO, "runs", "pbr_sr", "init")
SRC = {"r1": os.path.join(INIT, "r1"), "matup": os.path.join(INIT, "matup"),
       "matup_ft": os.path.join(INIT, "matup_ft")}


def read(path, grey=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit("missing %s - run its stage first (bash scripts/init.sh status)" % path)
    x = img.astype(np.float64) / (65535.0 if img.dtype == np.uint16 else 255.0)
    if x.ndim == 3:
        x = x[..., :3][..., ::-1]                         # BGR(A) -> RGB
        if grey:
            x = x.mean(axis=2)
    return x


def u8(x):
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(SRC))
    args = ap.parse_args()
    src = SRC[args.arm]
    out = os.path.join(INIT, "read_" + args.arm)
    os.makedirs(out, exist_ok=True)

    kd = read(os.path.join(src, "baseColor.png"))
    rough = read(os.path.join(src, "roughness.png"), grey=True)
    metal = read(os.path.join(src, "metallic.png"), grey=True)
    n = read(os.path.join(src, "normal.png"))
    ao = read(os.path.join(INIT, "lr", "ao_hit4.png"), grey=True)
    shapes = {k: v.shape[:2] for k, v in dict(kd=kd, rough=rough, metal=metal, n=n, ao=ao).items()}
    if len(set(shapes.values())) != 1:
        raise SystemExit("size mismatch: %s" % shapes)
    ks = np.stack([ao, rough, metal], axis=-1)

    for name, x in (("sr_kd_output.png", kd), ("sr_ks_output.png", ks), ("sr_n_output.png", n)):
        if not cv2.imwrite(os.path.join(out, name), u8(x)[..., ::-1]):
            raise SystemExit("cannot write %s" % name)
    meta = {"arm": args.arm, "source": os.path.relpath(src, REPO), "size": list(kd.shape[:2]),
            "ks": "R = lr/ao_hit4.png (1-AO), G = roughness, B = metallic",
            "ranges": {k: [float(v.min()), float(v.max())] for k, v in
                       dict(kd=kd, ao=ao, rough=rough, metal=metal, n=n).items()}}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print("-> %s" % os.path.relpath(out, REPO))


if __name__ == "__main__":
    main()
