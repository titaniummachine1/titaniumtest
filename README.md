# titanium-hub (`ti`)

The one tool for Titanium Ataxx work: SPRT gates between git refs, books, relabeling, **NNUE training (GPU on Modal)**,
merge checks, coordinator/worker. Binary name: `ti`. Lived in the engine repo as `crates/titanium-hub` (branch
`exp/ti-hub`, last commit 1927bf7) until 2026-10-31; this repo is now its only home. Never train or gate through
ad-hoc scripts in the engine repo: extend `ti`.

## Build / test

    cargo test --release          # must be green before every commit
    cargo build --release         # target/release/ti(.exe)

The engine is a **pinned git dependency** (`Cargo.toml`, origin/main `20c7acc`), so this repo builds without the
engine checked out next to it. Working on a local engine checkout? Create the gitignored `.cargo/config.toml`:

    [patch."https://github.com/titaniummachine1/titanium-ataxx"]
    titanium = { path = "C:/gitProjects/titanium-ataxx/crates/titanium" }

Bump the `rev` in `Cargo.toml` when the engine moves; record it in the commit message.

## Layout

    src/main.rs   subcommand dispatch (sprt, regret, gate, log, merge-check, train, relabel, analyze, harvest, coordinator, worker)
    src/gate.rs   commit-vs-commit and net-vs-net gates, one record per run (docs/GATES.jsonl)
    src/sprt.rs   pentanomial SPRT, matches training/gate_summary.py
    src/games.rs  replay / explain / harvest of game logs
    src/book.rs   book checks (refuses the poisoned one-stone-start book)
    src/train.rs  NNUE trainer (TNN1/2/3 IO, CPU backend; CUDA backend + Modal runner in progress)
    src/hash.rs   dependency-free SHA-256, the project hash convention (first 8 hex digits, upper case)
    docs/REGRET.md   `ti regret`: deep-reference regret of a shallow search on fixed positions
    docs/HUB_TRAINING_PLAN.md   the mandatory training work list (H0-H10) and acceptance criteria

## Status of training (read docs/HUB_TRAINING_PLAN.md)

H0 done (repo settled), H1 done (TNN3 round trip of the main net is byte-identical). Everything else is open.
