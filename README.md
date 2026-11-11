# titanium-hub (`ti`)

The all-in-one Rust toolchain for [Titanium Ataxx](https://github.com/titaniummachine1/titanium-ataxx): clean-commit SPRT gates, SPSA tuning, game/bucket analysis, dataset packing, relabeling, CPU/CUDA NNUE & policy-head training, Modal GPU runs, Botzone packaging/ingestion, and distributed coordinator/worker execution.

Moved out of `crates/titanium-hub` into its own repository on 2026-10-31. Binary name: `ti`.

## Build & Test

```sh
cargo test --release            # run full suite (must be green before committing)
cargo build --release           # outputs target/release/ti (or ti.exe)
```

Optional Bullet GPU features:

```sh
cargo check --features gpu      # type-check GPU backend without CUDA toolkit
cargo build --release --features cuda
```

### Engine Dependency & Local Overrides

`Cargo.toml` pins the `titanium` crate from `titanium-ataxx` via git. For local engine development, add a gitignored `.cargo/config.toml`:

```toml
[patch."https://github.com/titaniummachine1/titanium-ataxx"]
titanium = { path = "C:/gitProjects/titanium-ataxx/crates/titanium" }
```

## Subcommands

### Gating, Verification & Tuning
- `ti gate` — builds clean commits from git (`HEAD`/`--test` vs `--base main`) into `_gate/exes/`, runs paired pentanomial SPRT (`--nodes` or `--time-ms`), checks dirty/commit banners + bundled signature, records results in `docs/GATES.jsonl`, and tags `gate/<run_id>`. Supports `--net-test`, `--test-args`, `--base-args`, `--book0`, and `--non-main-ab`.
- `ti sprt` — multi-threaded pentanomial SPRT match runner (`--baseline` defaults to verified `main` champion).
- `ti log [--n 20]` — prints the recorded gate ledger from `docs/GATES.jsonl`.
- `ti merge-check <REF|--net FILE>` — verifies a passing gate record against current `main` / net before merging.
- `ti spsa` — paired multi-threaded SPSA tuner over runtime engine `--tune` knobs.

### Game Analysis & Datasets
- `ti insight <games.jsonl> [--by-bucket]` — paired value-lost diff, eval-level bias, phase breakdown, and per-16-bucket NNUE breakdown (`--by-bucket`).
- `ti analyze <games.jsonl>` — replays games and reports turning points, mate proofs, and mover disagreements.
- `ti harvest <games.jsonl> --out <file>` — extracts verified game positions + outcomes/evals as holdout rows.
- `ti data pack --npz <v9.npz> --out <stem>` — streams compressed v9 `.npz` data into fixed 48 B `.tdat` + `.kata` + `.aux` + `.manifest.json` (`ti data stats`, `check-targets`, `check-leak`).
- `ti relabel` — persistent multi-threaded local search relabeler writing chunked search labels.

### NNUE & Policy Training
- `ti train run --backend cpu|cuda` — NNUE trainer (`TNN1/2/3`, shared engine feature mapper + D4 augmentation, D162 exact mate targets, `--result-weight-phase`, warmup/cosine LR, provenance checks).
- `ti train check` — preflight validator (manifest row counts, `.tdat` sanity, parity, `ft/ftb` int16 range).
- `ti train export` — Texel calibration (`--calibrate`), closed-form ridge refit of the `TNN3` disagreement head, QA quantization check, engine parity check, and `.prov` emission.
- `ti train parity` / `ti train buckets` / `ti train probe` — engine `f32`/quantized parity, per-bucket MSE/variance breakdown, and residual probing.
- `ti train shrink | rank | upscale | split | remap` — Net2Net width pruning, channel ranking, upscaling, bucket splitting, and bucket remapping.
- `ti train modal put|launch|status|fetch|stop` — manages Modal volume uploads and CUDA jobs via `modal/ti_modal.py` with cost caps and `docs/MODAL_RUNS.jsonl` logging.
- `ti policy` / `ti policy2` — fits move-ordering policy heads (`TPOL1/2/3`) on gate games + streamed Kata soft targets (multi-threaded with checkpointing/resume).

### Botzone & Distributed Cluster
- `ti botzone <match.json|dir|URL> --out <DIR>` — fetches/parses Botzone matches (with per-move stats) and exports D4-canonicalized `positions.bin` + `meta.tsv` for relabeling.
- `ti botzone-upload` — verifies package `MANIFEST.txt`, uploads `titanium_v2.so` over curl, verifies SHA-256 round-trip, and updates the Botzone bot version.
- `ti coordinator` / `ti worker` — HTTP job coordinator (with live dashboard) and worker clients for distributed SPRT/relabeling.

## Layout

```text
src/
  main.rs       CLI dispatch, local relabeler, coordinator, worker
  gate.rs       clean-commit SPRT gate builder, log & merge-check
  sprt.rs       pentanomial SPRT engine and verifier
  spsa.rs       SPSA search parameter tuner
  games.rs      game replay, analyze, insight (--by-bucket), harvest
  book.rs       opening book validation
  data.rs       v9 .npz -> .tdat/.kata/.aux packer & stats
  mapper.rs     engine-exact nnfeat mapping & D4 symmetry transforms
  targets.rs    D162 target builder (eval/outcome blend + proven mates)
  train.rs      TNN1/2/3 serialization, forward pass & sparse AdamW
  trainer.rs    CPU training loop, check, parity, buckets
  gpu.rs        Bullet CUDA trainer backend (--features gpu/cuda)
  export.rs     Texel calibration, disagreement-head refit & export
  probe.rs      NNUE weight-mass & residual analysis
  shrink.rs     shrink, rank, upscale, split & remap transforms
  policy.rs     TPOL1/2/3 policy head trainer + k3 pattern indexer
  policy2.rs    streaming Kata + game policy trainer with checkpoints
  modal.rs      Modal job launcher & cost-guard wrapper
  prov.rs       training/export provenance records & verification
  repro.rs      gate build reproducibility checks
  hash.rs       dependency-free SHA-256 helpers
  botzone.rs    Botzone match fetcher, analyzer & deduplicator
  bzupload.rs   Botzone package uploader & SHA verifier
  layout.rs     binary tensor layout helpers
  leak.rs       dataset train/val leakage checks
modal/
  ti_modal.py   Modal CUDA container image & entrypoint
docs/
  GATES.jsonl           append-only SPRT gate ledger
  MODAL_RUNS.jsonl      Modal launch log
  HUB_TRAINING_PLAN.md  NNUE training architecture & checklist (H0-H10)
  REVIEW_NOTES.md       engineering & training review notes
Dockerfile      containerized worker build
```
