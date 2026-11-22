"""The ONE Modal launcher for `ti` (H9). Modal's SDK is Python-only, so this file only builds the image and runs a
`ti` command line in it; all logic (training, export, parity, preflight) is in the Rust hub. A new job = new `ti`
arguments or a new `ti` subcommand, never a new script. Drive it with `ti train modal ...` (src/modal.rs), not by hand.

Cost rules (engine repo docs/runbooks/MODAL.md): one container, hard `timeout` around the command, GPU fallback list, $30/month cap.

Layout on the Volume `ti-data`:   /vol/data/*.tdat|.manifest.json   packed datasets (uploaded ONCE, `ti train modal put`)
                                  /vol/nets/*.tnn                    init nets
                                  /vol/runs/<tag>/                   outputs of a run (`ti train modal fetch --tag`)

Image: engine crate + nets -> dependency layer (hub Cargo.toml/lock + dummy main: bullet, flate2, CUDA build) -> hub src.
The dependency layer is cached by Modal until Cargo.toml/Cargo.lock/engine change, so a hub source edit rebuilds only `ti`.
The hub is built from a `git archive` snapshot of the committed HEAD (an edit during upload aborts the build; a run
must map to a git ref); uncommitted changes are NOT shipped (a warning is printed).
"""
import os

import modal

HUB_REPO = r"C:\gitProjects\titaniumtest"
ENGINE = r"C:\gitProjects\mainpromo-ataxx"  # D1520: back to main 33d6ea1 (the committed hub compiles against it); engine crate the hub is patched to locally (see .cargo/config.toml)
SNAP_ROOT = r"C:\gitProjects\logs\ti_snap"
PATCH = '[patch."https://github.com/titaniummachine1/titanium-ataxx"]\ntitanium = { path = "/src/engine/crates/titanium" }\n'
CPU = float(os.environ.get("TI_CPU", "4"))  # vCPUs of the container (loader threads); the loader may be the cap, see REVIEW_NOTES N4
RUSTFLAGS = "-C target-cpu=x86-64-v3 -L /usr/local/cuda/lib64/stubs"
BUILD = "cd /src/hub && CUDA_PATH=/usr/local/cuda RUSTFLAGS='%s' $HOME/.cargo/bin/cargo build --release --features cuda" % RUSTFLAGS
# CPU-only build: the CUDA binary links libcuda.so.1 and cannot even start on a CPU
# container, so game generation uses a second, CUDA-free binary. A separate target
# dir keeps both builds cached in the image layer.
BUILD_CPU = "cd /src/hub && CARGO_TARGET_DIR=/src/hub/target-cpu RUSTFLAGS='%s' $HOME/.cargo/bin/cargo build --release" % RUSTFLAGS


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
    # the current engine also include_bytes!s these two (config.rs pin test, policy.rs head)
    .add_local_file(ENGINE + r"\data\nnue\BUNDLE.txt", "/src/engine/data/nnue/BUNDLE.txt", copy=True)
    .add_local_file(ENGINE + r"\data\nnue\policy_main.bin", "/src/engine/data/nnue/policy_main.bin", copy=True)
    # D1520: the per-stage knob table is include_str!'d too since D1369
    .add_local_file(ENGINE + r"\data\nnue\knobs_main.txt", "/src/engine/data/nnue/knobs_main.txt", copy=True)
    # dependency layer: only the manifest + a dummy main, so bullet/CUDA/flate2 are compiled once and cached
    .add_local_file(HUB + r"\Cargo.toml", "/src/hub/Cargo.toml", copy=True)
    .add_local_file(HUB + r"\Cargo.lock", "/src/hub/Cargo.lock", copy=True)
    .add_local_file(HUB + r"\.cargo\config.toml", "/src/hub/.cargo/config.toml", copy=True)
    .run_commands(
        "mkdir -p /src/hub/src && echo 'fn main(){}' > /src/hub/src/main.rs",
        BUILD,
        "rm -rf /src/hub/src /src/hub/target/release/ti /src/hub/target/release/deps/ti-* /src/hub/target/release/.fingerprint/titanium-hub-*",
    )
    .add_local_dir(HUB + r"\src", "/src/hub/src", copy=True)
    .run_commands(BUILD + " && cp target/release/ti /usr/local/bin/ti-cuda" + " && " + BUILD_CPU + " && cp target-cpu/release/ti /usr/local/bin/ti")
)
vol = modal.Volume.from_name("ti-data", create_if_missing=True)
app = modal.App("ti-train", image=image)


def sh(cmd):
    import subprocess
    import time

    t = time.time()
    print("$ " + cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    print("[exit %d, %.1f s]" % (r.returncode, time.time() - t), flush=True)
    return r.returncode


# GPU fallback list: L4 queues sometimes (2026-10-31: 6+ min); the net is tiny so any of them is enough. Memory: the whole
# training .tdat (1.4 GB) lives in RAM (`--ram`), 16 GiB is plenty.
@app.function(gpu=["L4", "T4", "A10G"], cpu=CPU, memory=16384, timeout=3 * 3600, volumes={"/vol": vol})
def run(tag: str, args: str, timeout_min: int = 30) -> int:
    """Run `ti <args>` under a hard `timeout`; outputs belong in /vol/runs/<tag> (the caller puts --out there)."""
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc")
    os.makedirs("/vol/runs/%s" % tag, exist_ok=True)
    # GPU function: the CUDA build (the image keeps `ti` = CPU-only build for the gen containers)
    rc = sh("timeout %dm ti-cuda %s" % (timeout_min, args))
    if rc == 124:
        print("TIMEOUT: killed after %d min (cost cap)" % timeout_min)
    vol.commit()
    return rc


@app.local_entrypoint()
def main(tag: str, args: str, timeout_min: int = 30):
    rc = run.remote(tag, args, timeout_min)
    print("RUN %s exit code %d" % (tag, rc))
    raise SystemExit(rc)


# ---------------------------------------------------------------- game generation
# CPU-only: game generation needs no GPU, and the GPU function above would bill an
# L4/T4/A10G for pure CPU work. Measured locally: ~8.2 core-s per 100 ms game plus
# ~2.3 core-s to relabel it at 5k, i.e. ~7.3k games per dollar at Modal CPU rates.
GEN_CPU = float(os.environ.get("TI_GEN_CPU", "1"))
GEN_MEM = int(os.environ.get("TI_GEN_MEM_MB", "2048"))


@app.function(cpu=GEN_CPU, memory=GEN_MEM, timeout=4 * 3600, volumes={"/vol": vol}, max_containers=100)
def gen(tag: str, shard: int, games: int, time_ms: int = 100, timeout_min: int = 120, book0: int = 0) -> int:
    """One shard of fresh self-play games, written straight onto the volume."""
    os.makedirs("/vol/runs/%s" % tag, exist_ok=True)
    out = "/vol/runs/%s/shard%03d" % (tag, shard)
    # botzone rules are built into `ti gen` (bz_over/bz_counts); no flags needed
    cmd = (
        "timeout %dm ti gen --out %s.jsonl --games %d --time-ms %d "
        "--book /vol/data/book50k_std5k.txt --book0 %d --tt-bits 18"
    ) % (timeout_min, out, games, time_ms, book0)
    rc = sh(cmd)
    if rc == 124:
        print("TIMEOUT: shard %d killed after %d min (cost cap)" % (shard, timeout_min))
    vol.commit()
    return rc


@app.local_entrypoint()
def gen_main(tag: str, games: int = 250, shards: int = 20, time_ms: int = 100, timeout_min: int = 120):
    """Fan out `shards` CPU containers; each plays `games` games on its own book slice."""
    hs = [gen.spawn(tag, i, games, time_ms, timeout_min, i * games) for i in range(shards)]
    rcs = [h.get() for h in hs]
    bad = [r for r in rcs if r != 0]
    print("gen: %d shards, %d games each, exit codes %s" % (shards, games, rcs))
    raise SystemExit(0 if not bad else 1)




