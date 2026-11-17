#!/usr/bin/env python3
"""MatUp x4 of PBR-SR's 1024^2 LR maps, run as overlapping 128^2 tiles and blended.

    cd docker && docker compose run --rm matup python /workspace/variants/pbr_sr/init/matup_tiles.py

Why tiles. MatUp (external/matup_egsr, not edited) is built around 128^2 inputs
(--material_size 128, the paper's setting). Its SwinIR-L step upsamples the
renders without tiling: sr_swinir.py sets tile = None and pushes a batch of
whole images through the model. At 1024^2 that would not fit in memory. So the
LR maps are cut into 128^2 tiles, every tile is a MatUp run of its own, at the
size MatUp was made for, and the 512^2 results are blended back together.

Tiling (LR px; HR = x4):
  - The LR maps are reflect-padded by PAD = 8, so the image edge falls inside a
    tile's valid area.
  - Tiles are 128 at stride 112 (16 overlap), the last one anchored at the edge.
  - From each 512^2 output, CROP = 16 HR px are dropped on every side. MatUp leaves
    the outer tile_width//2 = 8 px as zero padding (data_utils.py: pad_output). Its
    bilinear init also wraps around the tile border (file_utils.read_img_as_float:
    BORDER_WRAP), which is wrong for a cut-out tile.
  - Neighbours then share 64 - 2*16 = 32 HR px, blended with a linear ramp.
  - A tile with no UV-mask texel in its kept area is not run. It, and anything
    outside every kept area, gets MatUp's own init: cv2 INTER_LINEAR x4.

MatUp's settings are its defaults (main.py), with --upscaler swinir as in its
README. Two inputs need a choice:
  - normal: no --ogl_normals. MatUp writes normals in its internal convention
    (data_utils.py: write_normal(..., opengl_normals=False)). Reading them in the
    same convention makes the output keep the input's convention, whatever
    Cradle_Globe's is. That is what PBR-SR's `read` init needs.
  - height: Cradle_Globe ships none, so it is a constant 0.5. MatUp displaces by
    disp_val*(h-0.5), so this means no displacement.
MatUp clamps roughness to >= 0.045 (dataLoader.load_roughness). That is its own
behaviour, kept.

Resumable: a tile with tiles/<y>_<x>/out/*/baseColor_opt.png is done. Rerunning
skips it, and --stitch-only only blends.

Written under runs/pbr_sr/init/matup/:
  tiles/<y>_<x>/...                    each MatUp run (SVBRDF in, out/<stamp>/ results, run.log)
  {baseColor,roughness,metallic,normal}.png   4096^2 16-bit, blended
  tiles.json                           the grid, which tiles ran, timings
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3))
MATUP = os.path.join(REPO, "external", "matup_egsr")
MAPS = ["baseColor", "roughness", "metallic", "normal", "height"]
OUT_MAPS = {"baseColor": True, "roughness": False, "metallic": False, "normal": True}  # name -> RGB?
S = 4            # MatUp's fixed scale
T = 128          # LR tile (MatUp --material_size)
STRIDE = 112
PAD = 8
CROP = 16        # HR px dropped per side
RAMP = 32        # HR px blend ramp (= the overlap left after cropping)


def read16(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit("cannot read %s" % path)
    if img.ndim == 3:
        img = img[..., ::-1]                                  # BGR -> RGB
    return img.astype(np.float64) / (65535.0 if img.dtype == np.uint16 else 255.0)


def write16(path, x):
    arr = np.rint(np.clip(x, 0, 1) * 65535).astype(np.uint16)
    if arr.ndim == 3:
        arr = arr[..., ::-1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, arr):
        raise SystemExit("cannot write %s" % path)


def starts(size):
    s = list(range(0, size - T + 1, STRIDE))
    if s[-1] != size - T:
        s.append(size - T)
    return s


def ramp_weights():
    """Weights over a tile's kept HR area: a linear ramp RAMP px wide on every side."""
    n = T * S - 2 * CROP
    r = np.minimum(np.arange(n) + 1, np.arange(n)[::-1] + 1) / RAMP
    r = np.clip(r, 0, 1)
    return np.outer(r, r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lr", default=os.path.join(REPO, "runs/pbr_sr/init/lr/lr16"))
    ap.add_argument("--mask", default=os.path.join(REPO, "runs/pbr_sr/init/lr/mask.png"))
    ap.add_argument("--out", default=os.path.join(REPO, "runs/pbr_sr/init/matup"))
    ap.add_argument("--limit", type=int, default=0, help="run at most N tiles (tests)")
    ap.add_argument("--stitch-only", action="store_true")
    args = ap.parse_args()

    lr = {m: read16(os.path.join(args.lr, m + ".png")) for m in MAPS}
    H, W = lr["baseColor"].shape[:2]
    pad = {m: cv2.copyMakeBorder(lr[m], PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101) for m in MAPS}
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) > 0
    assert mask.shape == (H * S, W * S), (mask.shape, H, W)

    grid = [(y, x) for y in starts(H + 2 * PAD) for x in starts(W + 2 * PAD)]
    tiles_dir = os.path.join(args.out, "tiles")
    log = {"grid": {"tile": T, "stride": STRIDE, "pad": PAD, "crop_hr": CROP, "ramp_hr": RAMP,
                    "lr_size": [H, W], "n": len(grid)}, "tiles": {}}

    def kept_area(y, x):
        """The tile's kept HR area in unpadded HR coordinates."""
        y0, x0 = (y - PAD) * S + CROP, (x - PAD) * S + CROP
        n = T * S - 2 * CROP
        return y0, x0, n

    def done(d):
        return sorted(glob.glob(os.path.join(d, "out", "*", "baseColor_opt.png")))

    ran = 0
    for (y, x) in grid:
        key = "%04d_%04d" % (y, x)
        d = os.path.join(tiles_dir, key)
        y0, x0, n = kept_area(y, x)
        m = mask[max(y0, 0):max(y0 + n, 0), max(x0, 0):max(x0 + n, 0)]
        if not m.any():
            log["tiles"][key] = {"status": "skipped (no UV texel)"}
            continue
        if done(d):
            log["tiles"][key] = {"status": "done"}
            continue
        if args.stitch_only or (args.limit and ran >= args.limit):
            log["tiles"][key] = {"status": "pending"}
            continue
        svbrdf = os.path.join(d, "SVBRDF")
        for mp in MAPS:
            write16(os.path.join(svbrdf, mp + ".png"), pad[mp][y:y + T, x:x + T])
        cmd = [sys.executable, "main.py", "--upscaler", "swinir",
               "--input_mat", svbrdf + "/",
               "--render_dir", os.path.join(d, "radiance_LR") + "/",
               "--upsamplings_dir", os.path.join(d, "radiance_Upscaled") + "/",
               "--output_folder", os.path.join(d, "out") + "/"]
        t0 = time.time()
        with open(os.path.join(d, "run.log"), "w") as f:
            code = subprocess.call(cmd, cwd=MATUP, stdout=f, stderr=subprocess.STDOUT)
        sec = round(time.time() - t0, 1)
        ok = code == 0 and bool(done(d))
        log["tiles"][key] = {"status": "done" if ok else "FAILED (exit %d)" % code, "seconds": sec}
        ran += 1
        print("tile %s  %s  %.0fs  (%d/%d)" % (key, log["tiles"][key]["status"], sec,
              sum(v["status"] == "done" for v in log["tiles"].values()), len(grid)), flush=True)
        if not ok:
            text = open(os.path.join(d, "run.log"), errors="replace").read()
            if "nvml error" in text or "CUDA error" in text or "out of memory" in text:
                raise SystemExit("GPU failure in tile %s - see %s" % (key, os.path.join(d, "run.log")))

    pending = [k for k, v in log["tiles"].items() if v["status"] != "done" and not v["status"].startswith("skipped")]
    with open(os.path.join(args.out, "tiles.json"), "w") as f:
        json.dump(log, f, indent=2)
    if pending:
        print("%d tile(s) not done yet: %s ... - rerun to continue" % (len(pending), pending[:5]))
        if not args.stitch_only:
            sys.exit(2)

    # ---- blend
    w_tile = ramp_weights()
    for mp, rgb in OUT_MAPS.items():
        base = cv2.resize(lr[mp], (W * S, H * S), interpolation=cv2.INTER_LINEAR)   # MatUp's own init
        acc = np.zeros_like(base)
        wsum = np.zeros(base.shape[:2])
        for (y, x) in grid:
            key = "%04d_%04d" % (y, x)
            res = done(os.path.join(tiles_dir, key))
            if not res:
                continue
            img = read16(os.path.join(os.path.dirname(res[-1]), mp + "_opt.png"))
            if not rgb and img.ndim == 3:
                img = img[..., 0]
            img = img[CROP:T * S - CROP, CROP:T * S - CROP]
            y0, x0, n = kept_area(y, x)
            ys, xs = max(y0, 0), max(x0, 0)
            ye, xe = min(y0 + n, H * S), min(x0 + n, W * S)
            if ye <= ys or xe <= xs:
                continue
            sub = img[ys - y0:ye - y0, xs - x0:xe - x0]
            ww = w_tile[ys - y0:ye - y0, xs - x0:xe - x0]
            acc[ys:ye, xs:xe] += sub * (ww[..., None] if rgb else ww)
            wsum[ys:ye, xs:xe] += ww
        covered = wsum > 0
        outimg = base.copy()
        div = wsum[covered][:, None] if rgb else wsum[covered]
        outimg[covered] = acc[covered] / div
        write16(os.path.join(args.out, mp + ".png"), outimg)
        log.setdefault("coverage", {})[mp] = float(covered.mean())
        print("blended %s  coverage %.3f (rest = MatUp's bilinear init)" % (mp, covered.mean()))
    with open(os.path.join(args.out, "tiles.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
