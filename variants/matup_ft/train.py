#!/usr/bin/env python3
"""Fine-tune MatUp's SwinIR-L on MatSynth with KAIR's own training script, unmodified.

    cd docker && docker compose run --rm matup python /workspace/variants/matup_ft/train.py
    (scripts/matup_ft.sh train runs exactly this, detached)

external/KAIR/main_train_psnr.py ("model": "plain") runs as __main__ via runpy. It
brings the official loop, L1 loss, EMA, scheduler and saving. Around it this file
does what TexSR-comp1's variants/sr_finetune/swinir/kair_train.py did, plus one
thing for the data:

1. STARTING WEIGHTS. KAIR ignores pretrained_netG: an empty models/ means random
   weights. So before the first run, models/0_G.pth and 0_E.pth are written from
   MatUp's own file, params_ema of
   external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth.
   That is the weight MatUp loads (sr_swinir.py), so step 0 is MatUp as released.
   A folder that already holds a checkpoint is a resume and is left alone.
2. AN END. KAIR's loop never ends; it only saves. define_Model is wrapped so the
   model saves as usual and then exits 0 at step >= --total-iter.
3. THE FIVE MAPS. KAIR's option parser runs os.path.expanduser on dataroot_H/L, so
   it accepts only a string, although its dataset (utils_image.get_image_paths)
   takes a list. utils_option.parse is therefore wrapped. After the official parse,
   dataroot_H / dataroot_L become the five <pairs>/<map>/HR and LRx4 folders, in
   the same order. The names inside are identical, and KAIR pairs by sorted
   position (data/dataset_sr.py), so every LR meets its own HR. No data is copied.

Resume: run the same command again. KAIR continues from the highest <N>_G/E/optimizerG
in models/, and the learning rate is recomputed from the step.
"""
import argparse
import glob
import json
import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 2))
KAIR = os.path.join(REPO, "external", "KAIR")
MATUP_W = os.path.join(REPO, "external", "matup_egsr", "pretrained_models",
                       "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")
MAPS = ["baseColor", "normal", "roughness", "metallic", "height"]


def load_options(path):
    """KAIR's comment rule: drop everything after // on each line."""
    with open(path) as f:
        return json.loads("".join(line.split("//")[0] + "\n" for line in f))


def apply_set(opt, items):
    for item in items:
        key, value = item.split("=", 1)
        node, parts = opt, key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise SystemExit("[fatal] --set %s: no such key" % key)
        try:
            node[parts[-1]] = json.loads(value)
        except ValueError:
            node[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opt", default=os.path.join(REPO, "variants", "matup_ft", "swinirL_ft.json"))
    ap.add_argument("--pairs", default="/workspace/runs/deeptex/datasets/matsynth",
                    help="matsynth_make_pairs.py output: <pairs>/<map>/{HR,LRx4}/")
    ap.add_argument("--total-iter", type=int, default=40000)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = ap.parse_args()

    opt = load_options(args.opt)
    apply_set(opt, args.set)
    task, root = opt["task"], opt["path"]["root"]
    save_every = opt["train"]["checkpoint_save"]
    if args.total_iter % save_every:
        raise SystemExit("[fatal] --total-iter %d is not a multiple of checkpoint_save %d" % (args.total_iter, save_every))
    if opt["train"]["checkpoint_test"] <= args.total_iter:
        raise SystemExit("[fatal] checkpoint_test would be reached, and there is no test set")

    # the five folder pairs, checked before anything starts
    hr = [os.path.join(args.pairs, m, "HR") for m in MAPS]
    lr = [os.path.join(args.pairs, m, "LRx4") for m in MAPS]
    counts = {}
    for m, h, l in zip(MAPS, hr, lr):
        nh = sorted(os.path.basename(p) for p in glob.glob(os.path.join(h, "*.png")))
        nl = sorted(os.path.basename(p) for p in glob.glob(os.path.join(l, "*.png")))
        if not nh or nh != nl:
            raise SystemExit("[fatal] %s: HR %d / LRx4 %d files, names %s - run scripts/r1.sh pairs"
                             % (m, len(nh), len(nl), "match" if nh == nl else "DIFFER"))
        counts[m] = len(nh)
    print("[matup_ft] pairs per map: %s (total %d)" % (counts, sum(counts.values())), flush=True)

    import torch
    models_dir = os.path.join(root, task, "models")
    os.makedirs(models_dir, exist_ok=True)
    if not glob.glob(os.path.join(models_dir, "*_G.pth")):
        state = torch.load(MATUP_W, map_location="cpu")
        state = state["params_ema"] if "params_ema" in state else state
        for net in ("G", "E"):
            torch.save(state, os.path.join(models_dir, "0_%s.pth" % net))
        with open(os.path.join(models_dir, "0_init.json"), "w") as f:
            json.dump({"from": os.path.relpath(MATUP_W, REPO), "key": "params_ema",
                       "bytes": os.path.getsize(MATUP_W)}, f, indent=2)
        print("[matup_ft] step 0 = MatUp's SwinIR-L (params_ema) -> %s/0_{G,E}.pth" % models_dir, flush=True)

    run_opt = os.path.join(root, "_opt", task + ".json")
    os.makedirs(os.path.dirname(run_opt), exist_ok=True)
    with open(run_opt, "w") as f:
        json.dump(opt, f, indent=2)

    sys.path.insert(0, KAIR)
    os.chdir(root)

    import utils.utils_option as option
    official_parse = option.parse

    def parse(*a, **k):
        o = official_parse(*a, **k)
        o["datasets"]["train"]["dataroot_H"] = hr
        o["datasets"]["train"]["dataroot_L"] = lr
        return o
    option.parse = parse

    import models.select_model as select_model
    official_define = select_model.define_Model

    def define_Model(o):
        model = official_define(o)
        save = model.save

        def save_then_stop(iter_label):
            save(iter_label)
            if int(iter_label) >= args.total_iter:
                print("[matup_ft] saved step %d >= --total-iter %d: done" % (int(iter_label), args.total_iter), flush=True)
                sys.exit(0)
        model.save = save_then_stop
        return model
    select_model.define_Model = define_Model

    sys.argv = ["main_train_psnr.py", "--opt", run_opt]
    runpy.run_path(os.path.join(KAIR, "main_train_psnr.py"), run_name="__main__")


if __name__ == "__main__":
    main()
