"""Side-by-side 3D viewer for the PBR-SR results (GT, LR, each arm's init and final).

    python variants/pbr_sr/viewer/view.py              collect + serve + open the browser
    python variants/pbr_sr/viewer/view.py --port 8800  another port
    python variants/pbr_sr/viewer/view.py --no-browser only print the URL

Standard library only (no pip install). Works on Windows and Linux.

What it does:
  1. Finds every run under runs/pbr_sr/<RUN>/outputs/ (the experiment folder is the one
     holding logging.log and final_results_val/). Those paths are ~600 characters, over
     Windows' 260 limit, so they are opened with the \\\\?\\ prefix.
  2. Copies the meshes and textures the viewer needs to short paths under
     runs/pbr_sr/viewer/data/ (ignored by git; copied again only when a source changed).
  3. Parses logging.log for the render metrics (Val_set1 / Val_set2 PSNR, SSIM, LPIPS at
     step 0 = init and step 2000 = final) and the texture metrics vs GT (the table after
     "SR PBR -- Step:N"), and writes runs/pbr_sr/viewer/manifest.json.
  4. Serves the repo root on 127.0.0.1 (local only) and opens
     variants/pbr_sr/viewer/index.html, which loads the manifest, the meshes, the
     textures and PBR-SR's own environment map.
"""
import argparse
import functools
import http.server
import json
import os
import re
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs" / "pbr_sr"
OUT = RUNS / "viewer"
DATA = OUT / "data"

# run folder -> (short id, label of the final result, label of its init)
ARMS = {
    "cradle_globe_paper": ("paper", "PBR-SR (paper init)", "init: DiffBIR + bicubic"),
    "cradle_globe_paper_matup": ("matup", "PBR-SR + MatUp init", "init: MatUp"),
    "cradle_globe_paper_matup_ft": ("matup_ft", "PBR-SR + MatUp-ft init", "init: MatUp-ft"),
    "cradle_globe_paper_deeptex_renloss_21": ("deeptex", "PBR-SR + DeepTex-R21 init", "init: DeepTex-R21"),
    # ablation: paper init, SSIM weight 10 instead of 100 (same init as "paper": shown once)
    "cradle_globe_paper_ssim10": ("paper_ssim10", "PBR-SR (paper init, SSIM 10)", None),
}
MAPS = ("kd", "ks", "n")          # texture_kd.png, texture_ks.png, texture_n.png
ROWS = ("Base color", "Roughness", "Metallic", "Occlusion", "Normal")


def lp(p):
    """Path usable past 260 characters on Windows."""
    p = os.path.abspath(str(p))
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + p
    return p


def find_exp(run_dir):
    """Experiment folder of a run: holds logging.log; prefer one with final results, newest."""
    found = []
    for root, dirs, files in os.walk(lp(run_dir / "outputs")):
        if "logging.log" in files:
            final = os.path.isfile(os.path.join(root, "final_results_val", "mesh", "texture_kd.png"))
            found.append((final, os.path.getmtime(os.path.join(root, "logging.log")), root))
            dirs.clear()
    return max(found)[2] if found else None


def copy(src, dst):
    dst = Path(dst)
    if dst.exists():
        s, d = os.stat(lp(src)), dst.stat()
        if s.st_size == d.st_size and s.st_mtime <= d.st_mtime:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lp(src), dst)


def png_size(path):
    with open(lp(path), "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big")


def parse_log(path):
    """{step: {"render": {"set1": {...}, "set2": {...}}, "maps": {row: {...}}}}"""
    out = {}
    text = open(lp(path), encoding="utf-8", errors="replace").read().replace("\r", "\n")
    lines = text.split("\n")
    rend = re.compile(r"(Val_set[12]): Env\.Lighting = (\S+), Renderings from STEP(\d+) PBR: (.*)")
    table_step = None
    for line in lines:
        m = rend.search(line)
        if m:
            vals = dict((k.lower(), float(v)) for k, v in re.findall(r"(\w+)=([-\d.]+)", m.group(4)))
            step = int(m.group(3))
            out.setdefault(step, {}).setdefault("render", {})[m.group(1)[4:]] = vals
            out[step]["light"] = m.group(2)
            continue
        m = re.search(r"SR PBR -- Step:(\d+)", line)
        if m:
            table_step = int(m.group(1))
            continue
        if table_step is not None:
            for row in ROWS:
                if line.startswith(row):
                    nums = [float(x) for x in line[len(row):].split()]
                    if len(nums) == 6:
                        out.setdefault(table_step, {}).setdefault("maps", {})[row] = {
                            "l1": nums[0], "mse": nums[1], "psnr": nums[2],
                            "l1_whole": nums[3], "mse_whole": nums[4], "psnr_whole": nums[5]}
    return out


def find_env():
    """PBR-SR's validation light, mud_road_puresky_4k (or any .hdr in the submodule)."""
    hdrs = sorted((REPO / "external" / "PBR-SR").rglob("*.hdr"))
    for h in hdrs:
        if "mud_road_puresky" in h.name:
            return h
    return hdrs[0] if hdrs else None


def collect():
    panels, meshes = [], {}
    shared_done = False
    for run, (aid, final_label, init_label) in ARMS.items():
        run_dir = RUNS / run
        if not run_dir.is_dir():
            print(f"  skip {run}: not found")
            continue
        exp = find_exp(run_dir)
        if exp is None:
            print(f"  skip {run}: no logging.log under outputs/")
            continue
        metrics = parse_log(os.path.join(exp, "logging.log"))
        # the run's stdout log has the same lines; it fills what logging.log lacks
        # (logging.log can miss the last table if the files were taken before the run exited)
        stdout_log = RUNS / "logs" / f"{run}.log"
        if stdout_log.is_file():
            for step, d in parse_log(stdout_log).items():
                for k, v in d.items():
                    metrics.setdefault(step, {}).setdefault(k, v)
        stages = [("init", "init_mehs", init_label, 0), ("final", os.path.join("final_results_val", "mesh"), final_label, 2000)]
        if init_label is None:   # init identical to another arm's
            stages = stages[1:]
        if not shared_done:   # GT and LR are the same in every run: take them once
            stages = [("gt", "gt_mehs", "GT", None), ("lr", "lr_mehs", "LR input", None)] + stages
        for stage, sub, label, step in stages:
            src = os.path.join(exp, sub)
            if not os.path.isfile(os.path.join(src, "texture_kd.png")):
                print(f"  skip {run} {stage}: no textures")
                continue
            pid = stage if stage in ("gt", "lr") else f"{aid}_{stage}"
            for m in MAPS:
                copy(os.path.join(src, f"texture_{m}.png"), DATA / pid / f"{m}.png")
            copy(os.path.join(src, "mesh.obj"), DATA / pid / "mesh.obj")
            meshes[pid] = os.path.getsize(DATA / pid / "mesh.obj")
            p = {"id": pid, "label": label, "group": stage if stage in ("gt", "lr") else aid,
                 "stage": stage, "res": png_size(os.path.join(src, "texture_kd.png")),
                 "maps": {m: f"data/{pid}/{m}.png" for m in MAPS}, "mesh": f"data/{pid}/mesh.obj"}
            if step is not None and step in metrics:
                p["metrics"] = metrics[step]
            panels.append(p)
            print(f"  {pid:18s} {label:28s} {p['res']}^2")
        shared_done = shared_done or any(p["id"] == "gt" for p in panels)
    env = find_env()
    manifest = {
        "panels": panels,
        "env": "/" + env.relative_to(REPO).as_posix() if env else None,
        "same_mesh": len(set(meshes.values())) == 1,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"  env map: {manifest['env']}")
    return manifest


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        """POST /__shot/<name>: the page's labelled screenshot -> runs/pbr_sr/viewer/shots/<name>.png"""
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", self.path[len("/__shot/"):])[:80]
        if not self.path.startswith("/__shot/") or not name:
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        (OUT / "shots").mkdir(parents=True, exist_ok=True)
        dst = OUT / "shots" / (name if name.endswith(".png") else name + ".png")
        dst.write_bytes(body)
        print(f"saved {dst.relative_to(REPO)}")
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):   # quiet: only errors
        if args and str(args[1])[:1] in "45":
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    print("collecting results ->", OUT.relative_to(REPO))
    man = collect()
    if not man["panels"]:
        sys.exit("no PBR-SR results under runs/pbr_sr/ - pull them first")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(Handler, directory=str(REPO)))
    url = f"http://127.0.0.1:{a.port}/variants/pbr_sr/viewer/index.html"
    print(f"\nviewer: {url}\n(Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
