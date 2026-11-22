# HUB TRAINING PLAN (2026-10-31, user order D224/D226)

Rule (user, 2026-10-31): **training runs through `ti train`, the standardized Rust tool we improve every day.
Python is not the training path.** `training/train_nnue.py` is frozen as a *reference oracle* for parity tests
only; once the hub passes the acceptance criteria below it moves to `training/attic/` with `modal_train.py`.

Everything below is read from code: hub `C:\gitProjects\hub-ataxx` (crates/titanium-hub, branch reads
`exp/refresh-blocked`, 0 external crates), engine `crates/titanium/src/nnue.rs` + `nnfeat.rs` (main `55a62ab`),
`training/train_nnue.py`, bullet `C:\gitProjects\bullet` @ `da426a4`.

## 1. What has to be trained (the main net, from its header)

`nnue_main.tnn` sha `7AE0122F` = `TNN3`, num_feats 26,196, W 256, L1 16, L2 32, NB 8, groups 55 (G1 G2 G3 G5 G6),
pairwise 0. Forward (train_nnue.py `NNUE`, engine `nnue.rs`):

    a_p = ft[feats(p)] summed + ftb                       (p = stm, ntm; ft shared)
    x   = clamp(concat(a_stm, a_ntm), 0, 1)               (2W = 512)
    h1  = clamp(l1[bucket] x, 0, 1)                       (16)
    h2  = clamp(l2[bucket] h1, 0, 1)                      (32)
    y   = out[bucket] h2 + 0.5 * (psqt_stm - psqt_ntm)[bucket]
    bucket = min(empties * 8 / 50, 7); cp = 400 y
    TNN3 extra: disagreement head dis_w[bucket] . (h1 || h2) + dis_b (ridge fit, no effect on y)

Recipe today: target 0.75*sigmoid(clip(cp,+-1500)/400) + 0.25*outcome, proven mates exact 1/0 (D162), loss
|sigmoid(y) - t|^2.5 weighted by wsum, Ranger (RAdam + Lookahead), EMA, cosine/auto LR, D4 augmentation on train.

## 2. Issues found (hub today)

| # | issue | consequence |
|---|---|---|
| I1 | `from_tnn_bytes` parses TNN1/TNN2 only (test `#[ignore]`d) | cannot warm-start from the main net |
| I2 | reads only dis_positions.bin + relabel chunks | no v9: no outcomes, no Kata rows, no wsum, no proofs |
| I3 | target = sigmoid(clip(cp)/400) only | no outcome blend, proofs squashed to 0.977 |
| I4 | no D4 augmentation | loses the 8x data the Python trainer gets |
| I5 | no validation set, no metrics beyond train loss | cannot see overfit or drift |
| I6 | no checkpoints / resume | a dead run restarts from zero |
| I7 | no export check (i16 range at QA 1024), no disagreement-head refit | TNN3 not producible |
| I8 | one global +-1.98 clamp, per-epoch cosine with 5% floor | no per-weight clip, no warmup/final LR |
| I9 | CPU only, no Modal runner | cannot train 29M rows in useful time |
| I10 | branch name exp/refresh-blocked vs docs exp/ti-hub; worktree state unverified | wrong-ref risk (fishtest rule) |
| I11 | no `ti test` coverage of training code | regressions invisible |

## 3. Design decision

One tool, two backends, everything else shared and owned by us:

    ti data pack      v9 npz  -> .tdat (fixed 48-byte records)           [Rust, flate2 + own npy reader]
    ti train check    preflight (refuses to launch on any red)           [Rust]
    ti train run      --backend cpu|cuda                                 [Rust]
                        cpu  = hub's native trainer (smoke runs, small fine-tunes, tests)
                        cuda = bullet_trainer graph (pinned da426a4) behind cargo feature `cuda`
    ti train modal    build + upload + launch + status + fetch on a Modal GPU (cost cap)
    ti train export   TNN3 writer + disagreement-head ridge refit + engine parity
    -> ti gate        (existing)

Shared Rust pieces (both backends use the same code, so CPU and GPU cannot drift):
- **mapper**: features from `titanium::nnfeat::both` (the engine's own function: zero feature drift), bucket rule,
  D4 = apply the symmetry to the bitboards, then compute features (exact, no perm table to keep in sync).
- **targets**: port of `make_targets` incl. D162 exact proofs, with the same self-test cases as
  `train_nnue.py --selftest-targets`.
- **TNN IO**: read/write TNN1/2/3, round-trip bit-exact against `nnue.rs::parse`.

Why bullet for GPU: it is Rust (lives inside `ti`, no Python), its graph has what our arch needs (`new_affine`
sparse inputs, `concat`, `crelu`, `select` by bucket, `power_error`, elementwise `mul` for wsum weights, per-weight
clipping, **Ranger** + RAdam + AdamW), our features come in through a mapper closure, and Zataxx's family uses it.
Missing in bullet: EMA (optional, port if the gate says it matters), a real CPU backend (only a mock device, so
local runs use the hub CPU backend).

## 4. Mandatory work items (in order; each ends with a test in `ti test`)

- **H0** housekeeping: settle the hub branch (exp/ti-hub), commit the worktree state, `ti test` green.
- **H1** TNN IO (fixes I1): parse/write TNN1/2/3; test: 7AE0122F read -> write is byte-identical; engine eval of a
  hub-loaded net == engine eval of the file on 10k positions.
- **H2** data pack (I2): `ti data pack` reads v9 npz (zip deflate via flate2, npy headers parsed by hand), writes
  `.tdat` records (occ0, occ1, blk, stm, titanium_cp, outcome, wsum, proof_kind, teacher_kind, cp5, zx_score) + a
  manifest with row counts and sha; test: counts and column checksums equal the npz.
- **H3** targets (I3): Rust `make_targets`; test: same cases as the Python self-test, and on 1M v9 rows the Rust
  targets equal the Python oracle's bit-for-bit (oracle run once, result file committed as fixture hash).
- **H4** mapper + D4 (I4): test: mapper indices == nnfeat::both; D4-transformed board's features == features of
  the transformed position; bucket rule matches nnue.rs.
- **H5** CPU backend to parity (I3-I8 on CPU): TNN3 arch forward/backward (v1 + psqt term), wsum-weighted
  power-2.5 loss, Ranger (+ AdamW option), warmup + cosine to final LR, per-weight clip, val split each epoch,
  checkpoints + resume. Tests: step-0 forward == engine `forward_f32` to 1e-5 (D130 parity law); tiny-overfit
  (1k rows -> loss < 1e-3); resume after N steps == uninterrupted run.
- **H6** metrics (I5): per epoch: val loss, mse_target, per-bucket loss, sign accuracy, **proven-row loss on a
  held-out proof slice** (1% of proven boards, split by board key, because v9 val has 0 proven rows).
- **H7** export (I7): TNN3 writer, ridge refit of the disagreement head in Rust (48 inputs x 8 buckets, closed
  form), QA-range report (|ft|, |acc| vs 32767/1024), engine parity on 10k positions.
- **H8** CUDA backend (I9): bullet graph of the same arch; warm start = load the TNN3 weights into the graph;
  test: step-0 output == hub CPU forward to 1e-4 on 10k positions; 1 superbatch loss matches CPU within noise.
- **H9** Modal runner (I9): image with CUDA devel + rust, build `ti --features cuda`, upload `.tdat` with sha
  check, `ti train modal launch|status|fetch|stop`, hard timeout + cost cap from docs/runbooks/MODAL.md, writes the
  DECISIONS/HANDOFFS run row itself (docs in real time).
- **H10** `ti train check` preflight (all of the above, refuses on red): data sha + counts, targets self-test,
  init net hash + parity, RAM headroom, Modal budget left, branch/commit clean (gate law 8).

## 5. Acceptance: "hub is ready for the v9 retrain"

1. `ti test` green incl. H1-H8 tests.
2. `ti train check` green on v9.
3. Warm start from 7AE0122F reproduces the engine eval (parity) on CPU and CUDA.
4. A short CUDA smoke run (e.g. 1% of v9, a few minutes on Modal) lowers val loss and exports a TNN3 that the
   engine loads, with parity.
Then: heads-up to the user, explicit go, the real run, `ti gate` >= 100 games per leg.

## 6. Recipe changes this implies (each must win a gate, laws 2-4)

- EMA is not in bullet: CPU backend keeps it; CUDA starts without it; port if the gate shows it matters.
- Hypergrad / auto-LR (Python-only extras) are dropped; cosine to final LR (bullet default, plan R2) replaces them.
- Weight clipping becomes real (plan R1).
The first hub-trained net is gated against main like any candidate; nothing merges on parity alone.

## 7. Not examined yet

bullet's quantised SavedFormat path for our engine (we keep float TNN3 + load-time QA, so not needed now);
bullet CUDA build on Modal (nvcc version); EMA impact; CPU backend throughput on this 4C/8T laptop (measure in H5).
