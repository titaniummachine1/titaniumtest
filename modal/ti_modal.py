"""Thin Modal launcher for `ti` (H9). Modal's SDK is Python-only, so this file only builds the image and runs `ti`
commands; all training logic is in the Rust hub. Cost rules (engine repo docs/MODAL.md): one container, L4,
hard timeout, $30/month cap.

    python -m modal run modal/ti_modal.py::bench      # GPU parity + throughput smoke on the small val file (no big upload)

The image compiles `ti --features cuda` once; Modal caches the layer until the sources change.
The hub is built from a `git archive` snapshot of the committed HEAD, never from the live folder: an edit during upload
aborts the build (2026-10-31), the live `.cargo/config.toml` points at a Windows path, and a run must map to a git ref.
"""
import os

import modal

HUB_REPO = r"C:\gitProjects\titaniumtest"
ENGINE = r"C:\gitProjects\hub-ataxx"  # engine crate the hub is patched to locally (see .cargo/config.toml)
PACKED = r"C:\gitProjects\titanium-ataxx\data\nnue\packed"
SNAP_ROOT = r"C:\gitProjects\logs\ti_snap"
PATCH = '[patch."https://github.com/titaniummachine1/titanium-ataxx"]\ntitanium = { path = "/src/engine/crates/titanium" }\n'


def _snapshot():
    import subprocess
    import tarfile

    sha = subprocess.check_output(["git", "-C", HUB_REPO, "rev-parse", "--short=8", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", HUB_REPO, "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        print("WARNING: uncommitted hub changes are NOT in this build:\n" + dirty)
    dst = os.path.join(SNAP_ROOT, sha)
    if not os.path.isdir(os.path.join(dst, "src")):
        os.makedirs(dst, exist_ok=True)
        tar = dst + ".tar"
        subprocess.check_call(["git", "-C", HUB_REPO, "archive", "-o", tar, "HEAD"])
        with tarfile.open(tar) as t:
            t.extractall(dst)
        os.remove(tar)
    os.makedirs(os.path.join(dst, ".cargo"), exist_ok=True)
    with open(os.path.join(dst, ".cargo", "config.toml"), "w", newline="\n") as f:
        f.write(PATCH)
    print("hub build source = %s (HEAD %s)" % (dst, sha))
    return dst


HUB = _snapshot() if modal.is_local() else "/unused"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "git", "ca-certificates")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    .add_local_dir(ENGINE + r"\crates\titanium", "/src/engine/crates/titanium", copy=True, ignore=["target"])
    .add_local_file(ENGINE + r"\data\nnue\nnue_main.tnn", "/src/engine/data/nnue/nnue_main.tnn", copy=True)
    .add_local_file(ENGINE + r"\data\nnue\fut_small.tsn", "/src/engine/data/nnue/fut_small.tsn", copy=True)
    .add_local_dir(HUB, "/src/hub", copy=True, ignore=["target"])
    .run_commands(
        "cd /src/hub && CUDA_PATH=/usr/local/cuda RUSTFLAGS='-C target-cpu=x86-64-v3 -L /usr/local/cuda/lib64/stubs' "
        "$HOME/.cargo/bin/cargo build --release --features cuda && cp target/release/ti /usr/local/bin/ti"
    )
    .add_local_file(PACKED + r"\val_v9.tdat", "/data/val_v9.tdat")
    .add_local_file(PACKED + r"\val_v9.manifest.json", "/data/val_v9.manifest.json")
)
app = modal.App("ti-train", image=image)


def sh(cmd):
    import subprocess
    import time

    t = time.time()
    print("$ " + cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    print("[exit %d, %.1f s]" % (r.returncode, time.time() - t), flush=True)
    return r.returncode


# GPU fallback list: L4 queues sometimes (2026-10-31: 6+ min); T4 is cheapest, the net is tiny. Modal tries them in order.
@app.function(gpu=["L4", "T4", "A10G"], cpu=4.0, memory=16384, timeout=40 * 60)
def bench():
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; grep -m1 'model name' /proc/cpuinfo")
    net = "/src/engine/data/nnue/nnue_main.tnn"
    if sh("ti train gpu-parity --init %s --tdat /data/val_v9 --n 512" % net) != 0:
        return "PARITY FAILED"
    sh("ti train check --data /data/val_v9 --val /data/val_v9 --init %s" % net)
    for threads in (4, 8):
        sh("ti train run --backend cuda --data /data/val_v9 --val /data/val_v9 --init %s --out /tmp/b%d "
           "--bs 16384 --steps 600 --eval-every 200 --lr 1e-4 --lr-min 1e-5 --warmup 50 --threads %d --val-rows 16384 --ram"
           % (net, threads, threads))
    return "done"


@app.local_entrypoint()
def main():
    print(bench.remote())
