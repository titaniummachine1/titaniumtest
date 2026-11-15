# `ti regret` — counterfactual regret against a deep reference

Measures how much VALUE a search configuration throws away on fixed positions, compared with a much
deeper search of the same positions. Built to replace (or pre-filter) game-result noise when judging
search knobs: a 100-game SPRT resolves a few Elo at best, while one position yields a regret number
with ~12 cp standard deviation, i.e. ~200 positions resolve ~1.6 cp/move.

Implemented in `src/main.rs` (`run_regret_cli`, dispatch arm `regret`). No Python.

## Usage

```powershell
ti regret --book data/nnue/book50k_std5k.txt --book0 33000 --limit 200 `
          --ref-nodes 500000 --nodes 10000 --threads 8 `
          --candidate "<exe> serve --tune pol_w=50,... --network <net>" `
          --baseline  "<exe> serve --network <net>" `
          --out logs/gates/regret_run.csv
```

| flag | default | meaning |
|---|---|---|
| `--book` | required | position file (49 cells + side), read with `book::load` |
| `--book0` | 0 | first book line index (wraps) |
| `--limit` | 0 = all | number of positions |
| `--ref` | `--baseline` command | engine used as the deep reference |
| `--ref-nodes` | 500000 | reference node budget |
| `--nodes` | 10000 | shallow probe node budget |
| `--threads` | auto | parallel positions; each thread owns its engine processes |
| `--candidate` | none | second shallow engine (optional); without it only the baseline is measured |
| `--out` | none | per-position CSV |

## Method

1. Reference search of the position at `--ref-nodes` → `(m_ref, v_ref)` (value from the mover's view).
2. Shallow search at `--nodes` → `(m_shallow, v_shallow)`.
3. Same move → regret 0. Different move → the child after `m_shallow` is searched by the reference at
   `--ref-nodes`; the child value is the opponent's view, so the played move is worth `-v_child` and
   `regret = v_ref + v_child`, clipped to ±2000 cp (clip count reported).
4. Reported: move agreement %, mean regret cp/move, mean value gap `v_ref - v_shallow`, mean depth,
   and a per-empties phase table (30-49 / 20-29 / 10-19 / 0-9).

Mate scores are mapped to `±(30000 - 100·|mate|)` so ordering is preserved without drowning cp
statistics.

Determinism: fixed node budgets, `uainewgame` before every search (clears TT/killers/history),
one engine process per thread, no clock involved.

## Measured example (2026-11-02, engine exe 3060E318, net 46FCB5A4)

200 book positions from 33000, ref 500k, probe 10k, 8 threads, 487 s:

```
cand : agreement 55.0% | regret 7.0 cp/move | value gap +2.2 cp | depth 8.2
base : agreement 57.5% | regret 8.6 cp/move | value gap +3.3 cp | depth 8.1
delta: regret -1.6 cp/move | agreement -2.5 pp | value gap -1.1 cp
```

Paired difference −1.58 cp/move, sd 12.1, SE 0.85 (t −1.84). Only 13 of 200 positions differ in
agreement (106 both agree with the reference, 81 neither) → the candidate's misses are cheaper, not
fewer. Full write-up: engine repo `docs/RESEARCH_REGRET_2026_11_02.md`, decision D383.

## Limits

- **Position coverage decides everything.** With the paired book, every position is an opening
  (30-49 empties here), so the tool says nothing about the endgame.
- The reference is the same engine at a bigger budget, not ground truth; regret is relative to it.
- Regret is a screening objective, never a promotion: the paired game gate decides (LAWS B1/B2).
- Never read it below ~100 positions (the first 8 positions of the measured run gave the opposite
  agreement sign).
- The child re-search starts from a cleared TT, so it is a fresh deep search, not a PV extension.
