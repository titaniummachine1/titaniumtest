# REVIEW NOTES for Opus (written by Sonnet 5.5, 2026-10-31)

User order: "leave notes where you are unsure so Opus can say how stupid those decisions were; we want to squeeze
everything out of our bare metal". Each item = what I did, why, what I am NOT sure about, how to check it.
Status key: VERIFIED (tested, number given) / UNSURE (judgement call) / UNTESTED (written, not run).

## Data (H2/H3)
1. `.tdat` is 48 B/row, `.kata` 16 B/row, `.aux` 12 B/row. Kata columns, titanium_cp_old/v8 and zx_nodes live in side files so
   nothing is lost; training reads only `.tdat`. UNSURE: whether a single wider record would be faster to read on the GPU box
   (extra seeks are not an issue today because only `.tdat` is read). VERIFIED: wrapping column sums equal numpy for all
   columns, train 28,990,531 rows and val 601,957 rows.
2. Targets (H3, `targets.rs`): copy of the Python rule. 0.75*sigmoid(clip(cp,+-1500)/400)+0.25*outcome, proofs exact 1/0.
   VERIFIED: max |rust - python| 1.2e-7 over all 29.6M rows, proven rows bit-exact. UNSURE (policy, not code): the 0.25 outcome
   blend and 0.75 are the OLD recipe; user prefers pure exact outcome for proofs (done) and I kept the blend for the rest.
   Teacher-stage `teacher_p3` is plumbed (`make_target(.., Some(p3))`) but no flag uses it yet.
3. Draw proofs are NOT exact: only |cp|>=98000 is treated as proof. A proven draw has cp ~ 0 and stays a blended target.
4. `wsum` is used as a raw loss weight, rows with wsum 0 are skipped (CPU) / weight 0 (GPU). UNSURE: Python normalised
   differently (its loss is a weighted mean per batch). CPU path divides the step by sum(wsum) of the batch; the GPU graph
   divides by batch size (bullet's gradient_factor is 1/bs). Same direction, slightly different effective LR when mean wsum != 1.
   CHECK: mean(wsum) of v9 and rescale lr if it is far from 1.

## Mapper / D4 (H4)
5. D4 is applied to the BITBOARDS and then the engine's own `nnfeat::both` computes features (no permutation table).
   VERIFIED: G1 features follow the square map for all 7 non-identity modes. UNSURE: G2 (best-attack anchor) and G5 tie-breaks may
   pick a different but equivalent anchor on a transformed board, so features of the transformed board can differ from
   `perm[mode][features]` (what Python did). I believe the transformed-board version is the CORRECT augmentation (it is a
   real position) but did not diff it against the Python table. Cost: board_of + features per sample, not a table lookup.
6. D4 mode on GPU comes from a hash of (occ0, occ1, step) so no RNG state is shared between mapper threads. UNSURE: hash
   quality (8 modes from 3 bits of a multiplicative hash). Cheap to replace.
7. Blockers are always 0 (v9 has no blocker column; the 4-corner start has none). If blocker positions ever matter the
   `.tdat` needs a column.

## CPU backend (H5)
8. AdamW, NOT Ranger, warmup + cosine, no EMA, lazy updates for untouched feature rows (no decay on them). This is a smoke /
   reference backend; the production recipe is the GPU backend. UNSURE: lazy sparse AdamW is not equivalent to dense AdamW
   (moments of untouched rows are frozen, not decayed). Fine for smoke, do not compare its curves with Python's.
9. Step-0 parity VERIFIED: hub `forward_y` == engine `forward_f32` exactly (max diff 0.0) on 19,456 real positions for the
   bundled net and `nnue_main.tnn`; quantised engine eval differs from f32 by mean 0.8 cp, max 11 cp (nnue_main).
10. Speed: 19.7k rows/s on 6 threads (4C/8T Haswell, f32, no SIMD intrinsics, autovectorised). Bare-metal upside left on the table:
    explicit AVX2 for the FT row accumulate and the backward row adds, batching samples per thread to reuse the l1 weights,
    avoiding the per-sample `Vec` allocations in `forward_backward_weighted` (there are ~10 per call). NOT done, easy win.
11. Resume restarts optimizer moments (CPU). GPU resume uses bullet's own checkpoint (weights + optimiser state).

## GPU backend (H8) — UNTESTED ON A GPU
12. `gpu.rs` type-checks only (`--features gpu`, mock device). Nothing ran on CUDA. The graph is:
    ft(26196->256) crelu on both perspectives, concat 512, l1 (512->16*8) select(bucket) crelu, l2 (16->32*8) select crelu,
    out (48->8) select, + 0.5*(psq(stm)-psq(ntm)).select(bucket); loss = wt * |sigmoid(y)-t|^2.5 summed, grad factor 1/bs.
13. WEIGHT LAYOUT is an assumption: I assume bullet stores an affine weight of shape (out,in) column-major
    (element (o,i) at i*out+o). ft and psqt then equal our feature-major order, the dense layers need `layout::to_col_major`.
    The round trip net->graph->net is tested to be the identity, but whether bullet's layout really is column-major is
    UNVERIFIED until `gpu-parity` (hub forward vs graph forward on the same net) runs on a GPU. If parity fails, flip the
    layout helpers first.
14. `psqb` (psqt bias) cancels in stm-ntm; it is trained but ignored on export. Harmless; costs 8 floats.
15. Ranger params: beta1 0.9 (bullet default is 0.99, nnue-pytorch uses 0.9), beta2 0.999, k 6, alpha 0.5, weight decay 0,
    clip +-1.98 on EVERY weight (Python only clipped ft/ftb to QA range; clipping l1/l2/out at 1.98 might hurt — their max |w| is
    0.86 in the current net so probably harmless, UNSURE for fresh training).
16. No EMA on GPU (bullet has none). Python used EMA 0.999 and I do not know which of live/EMA weights produced 7AE0122F.
    Lookahead gives a similar smoothing effect. UNSURE; test by A/B gate if the result is borderline.
17. Reader: endless random 1024-row blocks with replacement, 128-block shuffle windows. An "epoch" is not defined.
    UNSURE: sampling with replacement vs a true permutation; at 29M rows with ~1M-step runs the difference is small.
18. Validation runs on the CPU (`trainer::evaluate`) every superbatch after pulling weights to host: 65k rows ~ <1 s. On a GPU
    box with few cores this may take longer; watch the wall clock.
19. Data loading threads for the mapper (8) call `nnfeat::both` per sample: on a Modal GPU box with few vCPUs the CPU
    mapper, not the GPU, will be the bottleneck. This is the biggest open performance question; the fix would be a pre-computed
    feature cache or moving feature generation to a CUDA kernel. UNMEASURED.

## Process
20. Hub moved to its own repo at user order. It builds against the PUBLISHED engine (rev 20c7accb); a local gitignored
    `.cargo/config.toml` patches it to the engine worktree for development. Check `git diff` of Cargo.lock before pushing.

## Preflight (H10)
21. `ti train check` samples 16 evenly spread 1024-row blocks of train plus one val block, not the whole file (full scan of
    1.4 GB takes seconds and could be added). It does not prove the file equals the npz; `ti data pack` prints the wrapping
    sums for that, and the manifest stores them. A future `check --full` should recompute the sums from the `.tdat`.

## OPUS REVIEW (Opus 5.5, 2026-10-31) — verdicts, with measured numbers
Measured: mapper (D4 + engine features) 350k rows/s/core on this Haswell; avg 106 active features/perspective;
val wsum mean 1.215, median 1.0, max 37.7.
- 1 tdat layout: OK. On the GPU box read the whole 1.4 GB into RAM once instead of seeking per block (cheap, removes disk jitter).
- 2/3 targets + draws: OK, code is right; the 0.25 blend on non-proven rows is a RECIPE question for the user, not a bug.
- 4 wsum: not a bug (GPU effective LR x1.215 vs CPU). REAL CONCERN: max weight 37.7 lets one row dominate a 16k batch.
  Suggest an A/B with wsum clipped at ~8 before the first long run.
- 5 D4 on bitboards: KEEP, it is more correct than the Python permutation table and costs nothing measurable.
- 6/7: OK.
- 8 CPU lazy AdamW: dubious for real training, fine for smoke. CPU backend = smoke/parity only, never production.
- 10 CPU speed: irrelevant to money (training is on the GPU); only worth it if we decide to train locally for $0.
- 12/13 graph weight layout: THE biggest risk. Must be proven by a graph-vs-hub forward parity on the GPU before any run.
- 15 clip +-1.98 on all weights: harmless for the current net (max |dense w| 0.86). OK.
- 16 EMA: Sonnet was right to flag it. Python evaluated BOTH live and EMA weights every epoch and kept whichever was best
  on val (train_nnue.py ~749, `_ep%d_ema.pt`). The hub has no EMA, so it may lose that free gain. Cheap fix: keep a
  host-side average of the last k superbatch snapshots (SWA-style) and validate both, same choose-best rule.
- 17/18/21: OK.
- 19 mapper bottleneck: 350k rows/s/core means a 4-core box maps ~1.4M rows/s, about an epoch of v9 in ~21 s. Probably not
  the bottleneck; the Modal benchmark decides.
NEW (not in Sonnet's list):
- N1 the GPU superbatch callback writes a 28 MB .tnn plus a full bullet checkpoint every superbatch: keep last + best only.
- N2 money: the net is tiny (~90k MACs per sample forward), so GPU time per run should be minutes. The real cost risks
  are rebuilding the CUDA image on every launch, re-uploading data and idle or parallel containers. Bake `ti` into the
  image once (cached by rev), keep `.tdat` in a Modal Volume uploaded once, one L4 container, hard timeout.
- N3 (FOUND ON THE GPU, FIXED): note 12 was wrong. TNN3 output reads h2 only (out_in = l2 = 32), not h1||h2 (48); h1||h2 is
  TNN2. The graph asserted `v2` and panicked on the main net before any training; it now follows the net's own flag like the
  CPU forward. Lesson: the graph shape must come from the net header, never from the doc.
- N4 GPU RESULT (2026-10-31): note 13 (column-major layout) CONFIRMED: graph vs hub max diff 7.2e-7 on 512 rows. A10: 1.85M pos/s,
  same at 4 and 8 loader threads on 4 vCPU, so the loader may be the cap. Next measurement: cpu=8 to see if it scales.
  v9 epoch ~16 s, ~$0.006/epoch.

## Breathing-cycle plan (user idea 2026-10-31, Opus opinion, OPEN DECISION, nothing launched)
User: train the champion to saturation and/or upscale it to 512, train more, then Net2Net downscale (never from scratch);
afterwards speed work towards 400k-1M nps. Opus view:
- Do NOT upscale 7AE0122F: it IS the W=256 downscale of the W=512 ep7 net 8AB5B82E. Upscaling it back gives a 512 that only
  knows what survived the downscale. Start the grow phase from 8AB5B82E itself (+52 Elo/node vs champion, D151).
- Grow: continue 8AB5B82E on v9, small LR (Mistake 39), --ram --guard --ema, stop when val loss plateaus. If 512 saturates
  early, the next grow step is Net2Net 512->1024 (a teacher does not need to be fast).
- Shrink: the teacher -> 256 (or smaller for speed) needs DISTILLATION, not just a downscale. MISSING IN THE HUB: no `ti` Net2Net
  up/downscale (only training/net2net_*.py) and no distill mode (the trainer learns data targets, not teacher outputs).
- W=512 GPU throughput not measured yet (256 = 1.85M pos/s on A10).

## Overfit probes (added 2026-10-31 after user go "yes exactly as you said", UNCOMMITTED until smoke, UNTESTED on GPU)
- D4 VERIFIED ON in code: CPU `rng.below(8)` per sample, GPU per-visit hash of (occ, batch) -> mode 0..7; `--no-d4` turns it off.
- Each superbatch now also prints: train-probe MSE (16k fixed train rows) + val-train gap, val MSE under all 8 D4 modes
  (8k rows each) + spread (WARNING > 5%, threshold is a guess), probe time. The GPU idles during the callback: superbatch =
  eval_every (500) x 16384 = 8.2M rows ~ 4.4 s GPU at 256-wide, so the probes may cost a noticeable share. MEASURE IN SMOKE.
- `--holdout-mode k` (user idea): never train orientation k, report its val MSE vs the 7 trained ones. DIAGNOSTIC ONLY.
  LIMIT: our inits (8AB5B82E, 7AE0122F) already trained on all 8 orientations, so from them the held-out gap can only show
  FORGETTING, not failure to learn the symmetry. A true test needs a net that never saw mode k. Low value for this cycle.
- MEASURED 2026-10-31 (`ti data check-leak`, src/leak.rs): v9 val is NOT position-disjoint. Exact copies in train 10 rows
  (0.002%), but D4-EQUIVALENT copies 37,123 of 601,957 = 6.17% (12.4% at 8-15 stones, 5.4% at 24-31, 4.5% at 40-47: all phases,
  so it is rotated duplicates in the data, not just symmetric openings). Train itself repeats 3.77% of its positions up to D4.
  Since training applies D4, those val rows are positions the net trains on in every orientation -> val loss is optimistic and
  hides overfit. FIX: `--write-clean` wrote data/nnue/packed/val_v9_clean.tdat (564,834 rows) -> use `--val /vol/data/val_v9_clean`.
  STILL UNCHECKED (b): game-level split. v9 split is a contiguous index cut (assemble_v7_dataset.py:238, train 0..T, val T..N)
  inherited from v8; no game ids in the npz, so neighbouring positions of one game can sit on both sides. Treat val as a
  plateau detector, not an absolute quality number. If it matters: rebuild val from games never used in train (needs game ids).

## SMOKE RESULT 2026-10-31 (Modal T4, W=512 8AB5B82E, 1500 steps lr 1e-5, hub 343e423, ~100 s)
- Launcher path PROVEN end to end (image build from git snapshot, Volume mounts, timeout wrapper, outputs in /vol/runs/<tag>).
- 512-wide GPU throughput 278k pos/s on T4 (GPU assigned from the fallback list was a T4, not the A10 of the 256 bench): v9 epoch ~104 s.
  UNKNOWN whether T4 or the 4-vCPU loader is the cap; L4/A10G would tell (cost per epoch may still favour T4).
- Probes cost 0.4 s per 30 s superbatch (1.4%): fine. D4 val spread 1.3-1.4% (threshold 5%). Train-probe is HARDER than val
  (0.0244 vs 0.0176): different data mix, so the val-train "gap" is negative; watch its trend, not its sign.
- lr 1e-5 moved val by +-0.00002 in 1500 steps: safe but nearly inert. The real run needs a larger LR (guard 1.15 protects).
- Launch footgun: Git Bash rewrites `/data/..` arguments to `C:/Program Files/Git/data/..`; use `ti train modal put` or MSYS_NO_PATHCONV=1.

## grow512a/b + memorisation + buckets (2026-10-31 night) - see engine docs/DECISIONS.md D232
- grow512b: val never beat the init (0.017632 vs 0.017639); LR halving x3 did not recover val; saturated on generalisation.
- Memorisation test (100k slice, lr 1e-3, no D4): loss 0.0231 -> 0.00013. Trainer is correct. Orientation probe works (17000% spread without D4, 1.5% with).
- `ti train buckets`: bucket 0 (<=6 empties) is 14% of rows, 42% of loss, 48% of target variance unexplained. NOTE bucket index = min(empties*8/50,7): 0 is the ENDGAME, 7 the opening.
- UNSURE: whether the endgame loss matters for play (search solves those positions). Idea, UNTESTED: down-weight bucket 0 rows or train with them excluded and see if mid-game buckets improve (capacity freed) - decided by games only.

## 2026-10-31 Shrinking without losing fitness: research + measured refit (D236)

Research (sources): Lottery Ticket Hypothesis (arxiv 1803.03635; rewinding variant 1903.01611) = iterative prune + rewind/retrain beats one-shot;
Neuron Merging (arxiv 2010.13160) = fold pruned neurons into kept ones via the next layer; Net2Net (arxiv 1511.05641) = function-preserving resize;
ThiNet / Channel Pruning (arxiv 1707.06342, 1707.06168) = choose channels, then least-squares reconstruct the next layer's input;
DeepChess (arxiv 1711.09667) = distilled small nets match, from-scratch ones are much worse; Lc0 distillation thread talkchess p=787865.

What we already have = the ThiNet/"image-style" step: `ti train shrink --refit` (pivoted-Cholesky channel selection + L1 ridge least-squares refit, no training).
Held-out val MSE (original -> same channels pruned -> pruned+refit):
- from 512 (0.017934): W384 0.019230 -> 0.018303 (71.5% recovered); W256 0.021650 -> 0.019085 (69.0%); W192 0.022655 -> 0.019841 (59.6%)
- from 256 champion 7AE0122F (0.018213): W192 0.019737 -> 0.018654; W128 0.022763 -> 0.019795; W96 0.023270 -> 0.020577; W64 0.025539 -> 0.021357
No cliff down to W64 before any retraining. Effective rank: 512 needs ~384 channels for 95% variance, 256 needs ~192, no dead neurons.

Plan (not run): stepwise chain (256->192->128->96->64), each step refit start + short distillation run (teacher = saturated net outputs via the p3 column,
`--distill L` blend, proofs untouched), LTH-style rewind rather than one-shot. Pick the knee by (val mse, engine nps), then game-gate 2-3 widths.
Needs `ti train teach` + `--distill` (not built) and a user go for GPU runs. Optuna-style search = bounded loop of capped `ti train modal launch` runs, only worth it for students.
