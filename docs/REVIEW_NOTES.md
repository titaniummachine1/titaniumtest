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

## 2026-10-31 Ceiling gate result + champion upscale run (D237)
- Gate: saturated 256 (grow256a best, v9 data, dis-head refit sha 6061F2EB) vs champion 7AE0122F, same exe FC2AE776, std5k book, 5k nodes: Elo -102.0 +- 39.9, LLR -3.00 (FAIL). Penta LL=71 LD=6 DD+WL=65 WD=5 WW=23. Lower val MSE (0.0178 vs 0.0182) did NOT give strength: v9 data/labels (Zataxx proofs etc.) move the net away from what wins games. Val MSE is not a strength metric.
- New: `ti train upscale` (Net2Net wider, step-0 parity measured: 256->512 delta val mse -3.1e-8) and `ti train ... --keep-all` (keeps every ckpt_<step>.tnn) to gate a strength-vs-step curve.
- Run: upscaled champion (data/nnue/sat/champ_up512.tnn) trained on v9, every eval checkpoint kept; then gate checkpoints vs champion to find where Elo peaks (distillation target). The peak may be step 0.

## 2026-10-31 v9 label audit (`ti data stats`) + SOTA training review (D238)
- train_v9 28,990,531 rows: 99.7% carry a 5k-search eval (cp5k); on every non-proven row cp == cp5k exactly (mean |cp-cp5k| 0.0), so the 5k search IS the signal. Proven rows (|cp|>=98000) = 2,438,314 (8.4%): their cp is replaced by the mate score and the target becomes exact 1/0 (their own cp5k is ignored). Zataxx-proof rows (proof_kind 1,2) 2.38M; ti_clash only 5,104.
- Suspect (unproven): exact 1/0 on 8.4% of rows (train-proven loss stays ~0.14, i.e. the net cannot fit them) + new Zataxx proofs pull the net off what wins at 5k. Earlier soft-proof test (blend 0.9) was judged only by val MSE, which cannot see proofs, so it proved nothing about strength. Needs a game gate, not val.
- SOTA review: nnue-pytorch loss = lambda*sigmoid(search score) + (1-lambda)*game result, lambda schedulable start->end, lr x gamma per epoch (exponential decay), Ranger, WDL-skipping of misevaluated positions; early runs used lambda 1.0, 0.8 kept improving. bullet simple.rs: constant WDL 0.75, step LR 1e-3 gamma 0.1, "lots of elo in the wdl and lr schedulers". Ours: 0.75 score/0.25 outcome, cosine, plateau drops: same family. What we do NOT do: fine-tune stage on the cleanest/highest-depth subset at low LR, WDL-skip of positions where search and outcome disagree wildly.
- Checkpoint hunt: run up512a (champion Net2Net-upscaled to 512, v9, lr 3e-5) kept ckpt every 250 steps (249..3749); gating 249/999/1999/3749 vs 7AE0122F at 5k nodes (logs/gates/up512a_*.log).

## 2026-10-31 Plan to maximise Elo before distillation (D239) - status at time of writing
- Finding: val MSE improved while games lost (256/v9 -102 +-40). Target is not aligned with Elo; games decide, not val.
- Provisional (5k nodes, vs 7AE0122F): champion Net2Net-upscaled to 512 with ZERO training = +101 +-89 (60 g, still running); ckpt 249 of up512a = +101 +-67 (110 g). Width alone ~ +100 at fixed nodes; trained checkpoints must beat the step-0 net, not just the champion. Two gates ran concurrently by mistake (pgrep absent in git bash): fixed-node results unaffected, only slower.
- Running/launched: up512a ckpts 249/999/1999/3749 gate chain (logs/gates/up512a_*.out), step-0 control (ctrl_up512.out), Modal runs up512pb5 (proof-blend 0.5, done, ckpts on Volume runs/up512pb5) and up512pb0 (proof-blend 0, launched, runs/up512pb0). All: init champ_up512, v9, lr 3e-5, 2500 steps, --keep-all, eval-every 250.
- Next (in order): gate pb5/pb0 ckpts; score/outcome lambda sweep (0.5/0.5, 0.9/0.1, 1.0/0); final straws = skip rows where 5k score and outcome disagree, low-LR fine-tune on cleanest rows, 512-teacher relabel of weak buckets; zero-init 1-ply-lookahead input columns only if all fail. Optuna-style search over per-datatype weights needs per-type weights in the trainer (not built) and a game-gate objective (~$3, hours); deferred pending go.
- Distillation starts only after the 512 is saturated; best checkpoint from the Elo-vs-step curves is the target. ETA for first answers 1-1.5 h, blends 2.5-3 h, full saturation verdict ~1 day.

## 2026-10-31 Is the architecture optimal for alpha-beta? (research, nothing run)
- Info side is covered (NET_ANALYSIS sec 32): G1+G3 carry the value, G6 free to drop, G5 only opening, Zataxx windows add nothing (D186). So "new inputs" is low prior.
- Speed side is the open lever: the big net calls nnfeat::both(b, groups) per eval (nnue.rs:227/255/388), i.e. it rebuilds features each node; only the small futility net and sancta use lazy incremental accumulators (search.rs:898,1019). G2/G5/G6 are global per node and cannot be updated incrementally (NET_ANALYSIS line 218). Zataxx gets NPS 17-20x with 2,916 one-hot windows + finny table; Moonbird-style pruning only pays with a fast eval (line 515).
- Untested: SCReLU/pairwise activation (Zataxx, Viridithas use them), incremental G1+G3 only, dropping G5/G6, threat edges, width cut (autaxx: width importance 0.0), Moonbird null-move R / aspiration / history malus with the current eval.
- Proposed experiment grid at FIXED TIME (100 ms), not fixed nodes: A champion; B G1+G3 incremental accumulator, G2/G5/G6 dropped, same 256 net retrained; C B + SCReLU; D B at W=128. Each needs engine work (incremental accumulator) before training; training via ti. Not started, needs go.

## 2026-10-31 Corner / edge / centre-distance PST probe (user idea)
`ti train probe` gained 6 candidates (pst_us/them = sum of Chebyshev distance-from-centre per stone, corner_us/them, edge_us/them). Residual of champion 7AE0122F on val_v9_clean (60k rows): single-feature correlations with the residual all |r| <= 0.053 (pst_them -0.038 all rows / -0.053 endgame, corner_them -0.042, edge_them -0.039, own-side ones <= 0.015); ALL 26 candidates together held-out R2 per bucket 0.97 / 0.11 / -0.63 / 0.40 / -0.01 / 3.37 / 3.27 / 1.43 %. G1 is a full per-square table, so a PST is already representable. Verdict: nothing there (<=0.3% of residual variance from any one).

## 2026-10-31 Constraint (user): AVX2 default, AVX-512 also supported
Width/speed curve (nps per width + 100 ms gates of refit-shrunk nets) must be measured on BOTH tiers: local PC = AVX2 (Haswell); AVX-512 via Docker sbx CPU sandbox if cpuid shows avx512 (CPU work goes to Docker, not Modal), Modal GPU hosts only as fallback. A width is only a candidate if it does not regress either tier.

## 2026-10-31 NULL TEST + mandatory evals (D240)
- Null test: champion 7AE0122F with ONLY its disagreement head refit by `ti train export` on v9 (1,367 bytes changed, sha EDC7BEC6) vs the original, 5k nodes, 300 g: **+45.4 +- 41.6, LLR +0.93** (LL 32, LD 3, DD+WL 56, WD 12, WW 47). The dis head drives search pruning (search.rs:1103) and its refit alone is worth ~+45 at 5k nodes (not yet significant, needs extension + 100 ms).
- Consequence: every candidate gated so far carried this refit. Re-read relative to the re-exported champion: 512 upscale untrained +72 -> width worth only ~+25; v9-trained 256 -102 -> v9 training costs ~-150. Earlier attributions (D237/D239) are corrected by this.
- Mandatory evals: engine branch exp/uai-info (worktree C:\gitProjects\info-ataxx, f211f71, exe 94751C57, bundled net 7AE0122F) prints `info depth D score cp|mate N nodes N time T nps N` before every bestmove (moves unchanged). `ti sprt` now refuses an engine that prints no info line (probe search at spawn). Games jsonl now carry score/depth/nodes per move. Old main exe FC2AE776 is refused from now on.

## 2026-10-31 Auxiliary-head lever sweep at 5k nodes (user order: "saturate the aux heads, count levers before downscaling")
What the dis head is: linear head over h1||h2 per bucket, target |400*y - cp5k|/400 clipped 2000 cp, proven rows excluded; used by search at search.rs:1103-1110: vol_credit = child_dis*DIS_FUT_NUM>>7 (DIS_FUT_NUM 48), a +1 depth when child_dis > 300, and DIS_RFP_NUM 32 for RFP; plus MB_FUT_DIV. Null test: refit alone +45 +-42 (D240).
Levers (all trunk-free): L1 fit rows (200k default vs 2M); L2 ridge l2 (0.1 / 1 / 10); L3 target definition (current vs signed bias vs search-depth-weighted); L4 search constants that consume dis (DIS_FUT_NUM, DIS_RFP_NUM, the 300 threshold, MB_FUT_DIV) re-tuned for the new head's scale (SPSA branch exp/search-spsa-disagree exists); L5 output-layer calibration (out_w/out_b ridge on cp5k, no trunk change); L6 s57w / psqt refit; L7 combination. Gate each vs the original champion on the new exe (info mandatory), 5k nodes, SPSA-style stop only after >=100 g; winner confirmed at 100 ms.
Running: logs/gates/levers_dis.sh -> exports dis_{r2m_l1,r200k_l01,r200k_l10}.tnn then gates lever_<tag>_vs_orig_5k.log in order r200k_l1, r2m_l1, r200k_l01, r200k_l10.

## 2026-10-31 Stage order to maximise search strength (user: Texel-style tuning of eval + dis head + search) (D241)
1 aux-head levers (rows, ridge, target, combination) -> 2 eval calibration (Texel-style fit of output scale/offset and s57w on 5k labels + game results; no trunk change) and dis-head refit on the final trunk -> 3 capacity-vs-speed sweep (width fixed) -> 4 SPSA of search constants in cp units (RFP margins, DIS_FUT_NUM, DIS_RFP_NUM, the >300 threshold, LMR, MB_FUT_DIV) on the chosen net at 100 ms, info-mandatory build, seat-balanced (branch exp/search-spsa-disagree has the SPSA harness) -> 5 if the net is changed again (distil/retrain), repeat 2 and 4. Reason: every search margin is in eval units, so its optimum moves with eval scale, head quality and nps.

## 2026-10-31 Guard rails for Texel/SPSA tuning (user: 1000-5000 nodes, never 100/500; last time it lost Elo)
- Evidence of the failure mode in our history: D147 (2026-10-29): SPSA found a positive gradient for dis_sn_num=48 at FIXED NODES but the 100 ms gate showed -32 Elo (72W-87L-5D): the parameter weakened the SmallNet leaf filter, +15% heavy evals, -20% nps. Fixed-node tuning cannot see a cost-per-node change. D147 item 2: dis_lmr_num lost -8 at 5k and -15 at 100 ms.
- Rules: (1) tuning games use 1000-5000 nodes per move (never <=500: margins that matter at depth 7+ do not show at depth 4); (2) a parameter that changes how many heavy evals run (anything touching the SmallNet filter, futility, RFP) is tuned or at least verified at TIME (100 ms) because its cost is invisible in nodes; (3) paired books, seats balanced, evals recorded, min 100 g per gate, extend on straddle; (4) calibration labels are the existing 5k-search values, never shallow ones; (5) every tuned set is gated vs the untouched baseline before it counts, SPSA gradient alone never merges.

## Gate bar (user 2026-10-31)
Only need to prove a candidate is objectively better than main (law 10): CI lower bound on score > 0.5 or SPRT upper bound. Magnitude is not a criterion (not Stockfish-style Elo counting). Extend straddles; never rank variants by Elo against each other.
- sandbox tit-sw1 (Docker cloud, AMD EPYC 16 vCPU): cpuid avx2 yes, avx512 NO (2026-10-31) -> AVX-512 tier needs Modal GPU host
- AVX-512 tier gates (need Modal GPU host; Docker has no avx512): HELD until later, probably past midnight, Modal budget goes to GPU training first (user 2026-10-31). AVX2 gates run on Docker/local meanwhile.

## Dis-head refit lever: PASSES law-10 bar (2026-10-31)
vs main 7AE0122F, paired book, CI lower bound >0: r200k_l1 5k +45.4+-41.6 (300g); r2m_l1 5k +58.5+-38.2 (300g); r200k_l01 5k +64.4+-38.5 (300g); r2m_l1 100ms AVX2 (Docker) +39.6+-35.3 (370g, ongoing). I wrongly waited for SPRT upper bound (+10 Elo1); law 10 replaced it. Variants not ranked (differences inside noise). AVX-512 tier gate still owed (Modal, after midnight). Merge needs user approval naming branch+net.

## Standing law (user 2026-10-31): stack every law-10 gain into main one at a time (BEHAVIOR 35 in engine repo). First: nnue_main.tnn 0BFFB378, main 1d26a49 local.

## Promotion legs (user 2026-10-31): search/cost-per-node changes need 5k AND 100 ms passing (+200 ms when aimed at longer search); a clearly negative leg blocks the merge. See BEHAVIOR 35a.
- Eval-quality-only changes (same width/arch, same nps) may be proven at 5k nodes alone; width/speed changes need timed legs (user 2026-10-31).
- BEHAVIOR 35b: ahead = presumed pass until proven otherwise; extend unresolved gates first, fresh openings via --book0.
- Screening gates are NOT SPRTs unless run to a bound: every unresolved gate (LLR inside +-2.94) is continued on fresh openings (--book0) until bound or 2000 games, positives first (user 2026-10-31). logs/gates/resolve.sh is the stopgap.
- USER REJECTION 2026-10-31: fut80 (DIS_FUT_NUM 80; 5k -18.5, 100ms -53.7 LOS 0.3%) and rfp16 (DIS_RFP_NUM 16; 5k -16.2) rejected; removed from queues. fut24 5k +40.7 (pass), 100ms +5.8 (unresolved) still live.
- Label depth for Texel/calibration (user 2026-10-31 asked 5k or 20k): use existing 5k cp5k labels (99.7% of rows); 20k only as held-out subset check later (needs relabel); time labels are load-noisy so 100ms stays a verification gate only.
- rfp56 (DIS_RFP_NUM 56) 5k: +64.4+-34.1 (300g, LLR +2.09); 100ms leg pending; merge needs both (BEHAVIOR 35a).
- Search-depth ladder (user 2026-10-31: "tuning on wrong search is not gonna help"): deployed Botzone search is ~220k-740k nodes/move (depth 12-15 at 4s), 100ms ~ 20k nodes, 5k is a cheap screen only. Screen 5k -> confirm 100ms -> confirm deep leg (~300k nodes) before merge of search changes; final SPSA at >=20k nodes. A clearly negative rung blocks merge.

## Three-phase buckets + opening book (user 2026-10-31)
- User: "3 main phases that get their own distribution of buckets ... opening might benefit most from disabling D4 on it" and "up to 8 or 10 pieces on board its still opening book that can be memorized".
- Plan: non-uniform bucket map (opening = >=38 empties i.e. <=11 stones; midgame coarse; endgame fine since bucket 0 = 42% of loss, mse 0.10). Needs engine bucket table keyed by empties (nnue.rs lines 230/258/393 use empties*B/50 today). Boundaries on old edges (empties 7/13/19/25/32/38/44) keep exact split-init parity. Uniform 8->16 split (`ti train split`, built, parity-checked) needs NO engine change: cheap first test, then per-bucket Texel calibration (`ti train export --calibrate`).
- D4 off for opening rows in training: separate gate.
- Opening book for <=10 stones (start has 4 stones => ~6 plies): search every position once, deep, store best move; engine `treebook` generator exists. Position count NOT yet counted (do not promise size).
- Dis-constant screening so far: thr200 (5k +1.2, 100ms +1.2) and thr450 (5k -6.9) flat => dropped. rfp56 leads (5k +64, 100ms +30), fut24 mixed (5k +41, 100ms +6).

## Distillation rule: no chains (user 2026-10-31)
User: "every step of downscale must be from biggest best strongest net usually so we dont do deaf phone game where information is lost throughout the distillation".
- Every student (distilled or refit-shrunk) comes DIRECTLY from the single strongest teacher net (by gate, not by width). Never student -> student.
- Teacher now = merged 0BFFB378 (W=256); saturated 512 nets are not teachers until they beat it per node AND at 100 ms.
- Labels = 5k-node search values (+ teacher outputs as soft target). When the teacher improves, redistil all students from it.
- Speed/strength curve may be non-monotone (user: dip at ~400 knps could rise again at ~1.5 Mnps). Report "best found", not "optimal". Sample coarse extremes first (64..512), refit-shrunk sweep understates small widths, search constants retuned per width.
- BEHAVIOR 35e: 100/200 ms legs decide search-affecting merges; 5k is diagnosis only (user 2026-10-31). rfp56 dropped: 5k fresh-opening rerun -23.3+-27.2 (582g) vs screening +64.4 (300g) = noise. cal_r2m 5k +51.3+-29.8 (512g) pending 100ms. fut24 100ms +20.3+-14.5 (2000g, lower +5.8); 200ms queued.
- fut24 deep leg (300k nodes) cancelled: 10 games took 453 s on 14 threads (~3.5 h for 400 g). Longer-search check = 200 ms leg (~45k nodes) + a later ~1 s leg (user 2026-10-31: time legs decide). BEHAVIOR 35c deep rung amended accordingly.
- Botzone tier (user 2026-10-31): speed in logs (~80k nps at ~300k nodes/4s vs ~230k nps local AVX2) shows Botzone is NOT AVX-512; AVX2 is the gating tier for deployment. AVX-512 leg dropped from owed-before-upload (Modal stays for GPU). Core is ~3x slower than this PC => 100 ms here ~ 300 ms there; keep 100/200 ms legs.

## CORRECTION: opening tree is NOT small (2026-10-31, measured)
`treecount` (scratch cmd in logs/varsrc/info-ataxx, 8-symmetry dedup, from standard start): new positions by ply 1..7 = 5, 64, 877, 12,356, 148,210, 1,866,676, 21,004,826 (~x12-14 per ply; stopped at ply 7). Jumps add no stone, so "<=10 stones" is NOT ~6 plies. My earlier claim ("small tree, search every position once") was WRONG. A full book is infeasible; a SELECTIVE book (best few moves per node within a margin, like `treebook`, thousands of lines) with deep search on main lines is the feasible lever. Not built; needs user go.
Also: dataset per-empties table (ti data stats): proof labels cover only <=4 empties (65-100%), 5k search finds forced results only at <=4 empties (89% at 1, 30% at 2, 11% at 3); <5% of rows have >=38 empties; rows at 43-47 empties show 12-54%% 'proven' = unexplained, possible artifact.

## Botzone deploy 2026-10-31 (user ordered; login only in conversation)
Uploaded data/titanium_v2.so sha EC4B4BAD (28,492,848 B), main ba09f2b, embedded net 00C01A49 (dis-head refit + per-bucket calibration). Driver unchanged 9550316F. Verified by download hash match. Rollback: logs/botzone_v12_backup (3F9E10BA, net 7AE0122F). Not included: fut24 (200ms leg -13 early), buckets, book, width.

## USER ORDER 2026-10-31 (late): NO Botzone upload of the 16-bucket net until it is actually RETRAINED and gated
User: "dont upload it to botzone until you finish retrain of it actually and gate it while still merged to main as better". cal16 (CE7C88D4) = split + calibration only, NOT retrained. Main keeps it (merged by user order); Botzone stays on EC4B4BAD (cal 8-bucket, uploaded earlier) / V13 until a real retrained 16-bucket net passes time gates. Upload job for 3CBD889C was killed mid-transfer; storage hash being re-verified.
