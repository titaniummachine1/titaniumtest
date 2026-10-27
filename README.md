<div align="center">
  <h1>⚡ TITANIUMTEST</h1>
  <p><strong>Standalone Fishtest / Fishnet Coordinator, Worker & Multi-Threaded Pentanomial SPRT Framework for Ataxx</strong></p>
  <p>
    <a href="https://github.com/titaniummachine1/titanium-ataxx"><img src="https://img.shields.io/badge/Engine-Titanium_Ataxx-00b4d8?style=flat-square" alt="Titanium Engine"/></a>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Language-Rust_2021-orange?style=flat-square&logo=rust" alt="Rust"/></a>
    <a href="https://tests.stockfishchess.org/"><img src="https://img.shields.io/badge/Architecture-Fishtest_%2F_Fishnet-success?style=flat-square" alt="Fishtest Inspired"/></a>
  </p>
</div>

---

## Overview

**`titaniumtest`** is a zero-Python, single-binary distributed testing and compute framework for [Titanium Ataxx](https://github.com/titaniummachine1/titanium-ataxx), inspired directly by Stockfish's **Fishtest** coordinator and **Fishnet** worker architecture.

Instead of fragile ad-hoc scripts, `titaniumtest` compiles into a single self-contained Rust executable that runs in three modes:

1. **`titaniumtest sprt`** — High-throughput local multi-threaded UAI tournament runner with exact **Pentanomial SPRT** (`[LL, LD, DD+WL, WD, WW]`), continuous LLR stopping (`[-2.94, +2.94]`), Elo + 95% confidence intervals, and Line- of-Superiority (LOS).
2. **`titaniumtest coordinator`** — Lightweight HTTP Fishtest server & live HTML dashboard. Organizes SPRT verification jobs, distributes paired game batches to connected workers, aggregates pentanomial statistics in real time, and automatically halts jobs the moment an SPRT boundary is crossed.
3. **`titaniumtest worker`** — Stateless Fishnet compute daemon for local machines, Docker containers, or cloud instances (Modal / RunPod / vast.ai). Polls the coordinator over HTTP (`GET /api/lease`), plays paired opening batches across all CPU threads, and posts pentanomial deltas (`POST /api/report`).

---

## Architecture

```
                     ┌──────────────────────────────────────┐
                     │      titaniumtest coordinator        │
                     │  (PC / Host Server — Port 8080)      │
                     │  • Live HTML Dashboard (GET /)       │
                     │  • Job Queue & Lease API (/api/*)    │
                     │  • Pentanomial SPRT LLR Aggregator   │
                     └──────────┬───────────────┬───────────┘
                                │               │
                 HTTP JSON Lease│               │HTTP JSON Report
          ┌─────────────────────┴───┐       ┌───┴─────────────────────┐
          ▼                         ▼       ▼                         ▼
┌───────────────────┐     ┌───────────────────┐             ┌───────────────────┐
│ titaniumtest      │     │ titaniumtest      │             │ titaniumtest      │
│ worker (Local PC) │     │ worker (Docker)   │     ...     │ worker (Modal)    │
│ --threads 8       │     │ --threads 16      │             │ --threads 64      │
└───────────────────┘     └───────────────────┘             └───────────────────┘
```

---

## Quick Start

### Build

```bash
cargo build --release
```

Produces `target/release/titaniumtest` (`titaniumtest.exe` on Windows).

---

### 1. Local Multi-Threaded Pentanomial SPRT (`sprt`)

Run a head-to-head SPRT gate directly on your machine across `N` worker threads:

```bash
titaniumtest sprt \
  --candidate "./candidate/titanium-cli serve --network bundled" \
  --baseline  "./baseline/titanium-cli serve --network bundled" \
  --book      "./book100.txt" \
  --time-ms   100 \
  --threads   8 \
  --max-games 400 \
  --elo0      0.0 \
  --elo1      5.0
```

Example output:
```text
[SPRT] Starting 400 games (200 pairs) across 8 threads @ 100ms/move
[SPRT] H0=0.0 Elo, H1=5.0 Elo | Bounds: [-2.94, +2.94]
[32/400] +16 -11 =5 (57.81%) | Elo: +54.4 +- 62.1 | LOS: 95.7% | Penta: [1, 1, 8, 4, 2] | LLR: +0.39 [-2.94, +2.94]
```

---

### 2. Distributed Fishtest Coordinator (`coordinator`)

Start the coordinator on your main PC (organizes work and serves the live web dashboard on `http://127.0.0.1:8080`):

```bash
titaniumtest coordinator \
  --bind      0.0.0.0:8080 \
  --candidate "./candidate/titanium-cli serve --network bundled" \
  --baseline  "./baseline/titanium-cli serve --network bundled" \
  --book      "./book100.txt" \
  --time-ms   100 \
  --max-games 1000 \
  --batch-pairs 4 \
  --elo0      0.0 \
  --elo1      5.0
```

* **Live Web Dashboard:** Open `http://localhost:8080/` in any browser for an auto-refreshing Fishtest table showing W/L/D, Elo, LOS, Pentanomial distribution, and LLR status.
* **Submit Additional Jobs Live (`POST /api/submit`):**
  ```bash
  curl -X POST http://localhost:8080/api/submit \
    -H "Content-Type: application/json" \
    -d '{"name":"mate-pruning-gate","candidate_cmd":"./cand serve","baseline_cmd":"./base serve","time_ms":100,"max_games":400,"batch_pairs":4,"elo0":0.0,"elo1":5.0}'
  ```

---

### 3. Stateless Fishnet Worker (`worker`)

Point any local machine, Docker container, or cloud worker at your coordinator URL:

```bash
titaniumtest worker \
  --server  http://127.0.0.1:8080 \
  --worker-id docker-ryzen-01 \
  --threads 8
```

When all SPRT jobs finish or cross an LLR threshold, the worker idles cleanly and automatically picks up the next job submitted to the coordinator.

---

## Pentanomial SPRT Math

Every opening from the book is played as a **game pair** `(Game 1: Candidate=Black vs Baseline=White, Game 2: Baseline=Black vs Candidate=White)`. Each pair produces one of five outcomes in `[0.0, 0.25, 0.50, 0.75, 1.0]`:

| Bucket | Outcome | Pair Score |
| :--- | :--- | :---: |
| **`LL`** | Candidate loses both games | `0.00` |
| **`LD`** | Candidate loses one, draws one | `0.25` |
| **`DD+WL`** | Both games drawn or 1 win / 1 loss split | `0.50` |
| **`WD`** | Candidate wins one, draws one | `0.75` |
| **`WW`** | Candidate wins both games | `1.00` |

This eliminates opening bias and color bias while providing accurate variance estimation ($\sigma^2$) for Log-Likelihood Ratio (LLR) stopping with bounds $\text{LLR} \in [\ln(\beta/(1-\alpha)), \ln((1-\beta)/\alpha)] = [-2.944, +2.944]$ at $\alpha = \beta = 0.05$.
