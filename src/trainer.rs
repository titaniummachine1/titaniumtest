//! H5: `ti train run` — CPU backend. Reads `.tdat` (see data.rs), D4-augments through mapper.rs, targets from
//! targets.rs (D162 exact proofs), wsum-weighted |sigmoid(y)-t|^2.5 loss, AdamW with warmup + cosine,
//! validation (overall / proven / non-proven), TNN3 checkpoints, resume. The GPU backend (bullet) uses the same
//! reader, mapper and targets; this one is the reference and the small-run tool.
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::process::ExitCode;
use std::time::Instant;

use crate::data::{Rec, REC};
use crate::mapper;
use crate::targets::{self, target_of};
use crate::train::{BatchGradients, SparseAdamW, TrainableNet};
use titanium::nnue::{Nnue, BUNDLED_NNUE};

const BLOCK: usize = 1024;

/// `.tdat` reader. `open` streams from disk (16 GB home box); `open_ram` reads the whole file once (Modal: RAM is
/// ~$0.008/GiB-h, far cheaper than a GPU waiting on disk seeks).
pub struct Tdat {
    f: Option<File>,
    mem: Option<std::sync::Arc<Vec<u8>>>,
    pub rows: usize,
}

impl Tdat {
    pub fn open(stem_or_file: &str) -> Result<Tdat, String> {
        let p = if stem_or_file.ends_with(".tdat") { stem_or_file.to_string() } else { format!("{stem_or_file}.tdat") };
        let f = File::open(&p).map_err(|e| format!("{p}: {e}"))?;
        let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        if len % REC != 0 {
            return Err(format!("{p}: size {len} is not a multiple of {REC}"));
        }
        Ok(Tdat { f: Some(f), mem: None, rows: len / REC })
    }
    pub fn open_ram(stem_or_file: &str) -> Result<Tdat, String> {
        let p = if stem_or_file.ends_with(".tdat") { stem_or_file.to_string() } else { format!("{stem_or_file}.tdat") };
        let t = Instant::now();
        let b = std::fs::read(&p).map_err(|e| format!("{p}: {e}"))?;
        if b.len() % REC != 0 {
            return Err(format!("{p}: size {} is not a multiple of {REC}", b.len()));
        }
        println!("[ram] {p}: {} MB in {:.1} s", b.len() >> 20, t.elapsed().as_secs_f64());
        Ok(Tdat { f: None, rows: b.len() / REC, mem: Some(std::sync::Arc::new(b)) })
    }
    /// Cheap clone for loader threads: shares the RAM copy, reopens the file when streaming.
    pub fn share(&self, stem_or_file: &str) -> Result<Tdat, String> {
        match &self.mem {
            Some(m) => Ok(Tdat { f: None, mem: Some(m.clone()), rows: self.rows }),
            None => Tdat::open(stem_or_file),
        }
    }
    pub fn read_block(&mut self, start: usize, n: usize, out: &mut Vec<Rec>) -> Result<(), String> {
        let n = n.min(self.rows - start);
        if let Some(m) = &self.mem {
            out.extend(m[start * REC..(start + n) * REC].chunks_exact(REC).map(Rec::from_bytes));
            return Ok(());
        }
        let f = self.f.as_mut().unwrap();
        f.seek(SeekFrom::Start((start * REC) as u64)).map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; n * REC];
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        out.extend(buf.chunks_exact(REC).map(Rec::from_bytes));
        Ok(())
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct ValStats {
    pub rows: usize,
    pub mse: f64,
    pub mse_proven: f64,
    pub n_proven: usize,
    pub mse_other: f64,
}

/// Evenly spread validation rows (val is source-ordered, so a contiguous slice would be biased).
/// D1130 (user: "remove any way to disobey me accidentally"): how many `--train` rows a child-sampling trainer uses.
/// ALL of them by default; `--rows` below that is refused unless `--experiment NAME`, whose output must sit under an
/// `experiments` folder (and a standard run must not). Also refuses a run whose samples (~45 legal children per row,
/// kept 1/every) would not fit in 8 GB on the 16 GB box, naming the `--child-every` that fits: rows are never dropped.
pub fn whole_rows_or_experiment(tool: &str, tr: &Tdat, req: Option<usize>, exp: &Option<String>, out: &str, every: u64, sample_bytes: usize) -> Result<usize, String> {
    let under = out.replace('\\', "/").split('/').any(|c| c.eq_ignore_ascii_case("experiments"));
    match (exp, under) {
        (Some(n), false) => return Err(format!("{tool}: --experiment {n}: --out must be under a folder named 'experiments' (got {out})")),
        (None, true) => return Err(format!("{tool}: --out {out} is under 'experiments' but this is a standard run; pass --experiment NAME")),
        _ => {}
    }
    let rows = match (req, exp) {
        (None, _) => tr.rows,
        (Some(r), None) if r < tr.rows => {
            return Err(format!(
                "{tool}: --rows {r} is refused: the standard trains on EVERY row of --train ({} rows, user order D1130); a subset only with --experiment NAME",
                tr.rows
            ))
        }
        (Some(r), _) => r.min(tr.rows),
    };
    let est = rows as f64 * 45.0 / every.max(1) as f64 * sample_bytes as f64;
    if est > 8e9 {
        let need = (rows as f64 * 45.0 * sample_bytes as f64 / 8e9).ceil() as u64;
        return Err(format!(
            "{tool}: {rows} rows at --child-every {every} need ~{:.1} GB of samples (> 8 GB on the 16 GB box): use --child-every {need} or more (every row stays in; never fewer rows)",
            est / 1e9
        ));
    }
    match exp {
        Some(n) => println!("[{tool}] EXPERIMENT {n}: NOT the standard ({rows} of {} rows)", tr.rows),
        None => println!("[{tool}] STANDARD (D1130): every row of --train = {rows} rows"),
    }
    Ok(rows)
}

pub fn load_val(val: &mut Tdat, rows: usize) -> Result<Vec<Rec>, String> {
    let blocks = (rows / BLOCK).max(1);
    let nblk = val.rows / BLOCK;
    let mut v = Vec::with_capacity(blocks * BLOCK);
    for i in 0..blocks.min(nblk.max(1)) {
        val.read_block((i * nblk / blocks) * BLOCK, BLOCK, &mut v)?;
    }
    Ok(v)
}

pub fn evaluate(net: &TrainableNet, val: &[Rec], threads: usize) -> ValStats {
    evaluate_mode(net, val, threads, 0)
}

/// Orientation-overfit probe: val MSE of the same rows under each of the 8 D4 modes (strided subset). A net that only
/// knows the stored orientation shows a large max-min spread; a net trained with D4 stays flat. Returns (min, max).
pub fn d4_modes(net: &TrainableNet, val: &[Rec], threads: usize, rows: usize) -> Vec<f64> {
    let step = (val.len() / rows.max(1)).max(1);
    let sub: Vec<Rec> = val.iter().step_by(step).copied().collect();
    (0..8).map(|mode| evaluate_mode(net, &sub, threads, mode).mse).collect()
}

pub fn d4_spread(net: &TrainableNet, val: &[Rec], threads: usize, rows: usize) -> (f64, f64) {
    let m = d4_modes(net, val, threads, rows);
    (m.iter().cloned().fold(f64::MAX, f64::min), m.iter().cloned().fold(0.0, f64::max))
}

pub fn evaluate_mode(net: &TrainableNet, val: &[Rec], threads: usize, mode: usize) -> ValStats {
    let part = (val.len() + threads - 1) / threads;
    let res: Vec<(f64, f64, usize, f64)> = std::thread::scope(|s| {
        val.chunks(part.max(1))
            .map(|ch| {
                s.spawn(move || {
                    let (mut a, mut p, mut np, mut o) = (0f64, 0f64, 0usize, 0f64);
                    for r in ch {
                        let t = target_of(r);
                        let y = net.forward_y(&mapper::board_of(r, mode));
                        let e = (1.0 / (1.0 + (-y as f64).exp()) - t.target as f64).powi(2);
                        a += e;
                        if t.proven >= 0 {
                            p += e;
                            np += 1;
                        } else {
                            o += e;
                        }
                    }
                    (a, p, np, o)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });
    let (a, p, np, o) = res.iter().fold((0.0, 0.0, 0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2, x.3 + y.3));
    let n = val.len();
    ValStats { rows: n, mse: a / n as f64, mse_proven: if np > 0 { p / np as f64 } else { 0.0 }, n_proven: np, mse_other: o / (n - np).max(1) as f64 }
}

fn lr_at(step: usize, total: usize, lr: f32, lr_min: f32, warmup: usize) -> f32 {
    if step < warmup {
        return lr * (step + 1) as f32 / warmup as f32;
    }
    let p = (step - warmup) as f32 / (total.saturating_sub(warmup)).max(1) as f32;
    lr_min + 0.5 * (lr - lr_min) * (1.0 + (std::f32::consts::PI * p.min(1.0)).cos())
}

fn tensor_report(n: &TrainableNet) -> String {
    let m = |v: &Vec<f32>| v.iter().fold(0f32, |a, &x| a.max(x.abs()));
    format!("max|w| ft {:.3} ftb {:.3} psqt {:.3} l1 {:.3} l2 {:.3} out {:.3}", m(&n.ft), m(&n.ftb), m(&n.psqt), m(&n.l1_w), m(&n.l2_w), m(&n.out_w))
}

pub fn run_train(args: &[String]) -> ExitCode {
    match run_train_inner(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("train run: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_train_inner(args: &[String]) -> Result<(), String> {
    let f = |n: &str| crate::flag_str(args, n);
    let d4 = !crate::has_flag(args, "--no-d4");
    let d4_rand = crate::has_flag(args, "--d4-rand");
    let holdout = crate::flag_u64(args, "--holdout-mode").map(|v| v as usize);
    let d4_all = d4 && !d4_rand && holdout.is_none();
    if crate::has_flag(args, "--d4-all") && !d4_all {
        return Err("--d4-all expands all 8 D4 orientations and conflicts with --no-d4 / --d4-rand / --holdout-mode".into());
    }
    let exp = f("--experiment");
    if exp.is_none() && (d4_rand || !d4) {
        return Err("--d4-rand and --no-d4 are forbidden on standard training (LAWS A13/D1631: default feeds all 8 D4 orientations per epoch, scrambled); pass --experiment NAME under experiments/".into());
    }
    if f("--backend").as_deref() == Some("cuda") {
        // LAWS D5 / D1159 / D1260 / D1554 / D1629 + LAWS A13: combined Muon + Schedule-Free AdamW (`--opt muon`)
        // is the MANDATORY default GPU optimiser; standalone `--opt sfadamw` or `--opt ranger` requires `--experiment NAME`.
        let opt_name = f("--opt").unwrap_or_else(|| "muon".into());
        if !matches!(opt_name.as_str(), "muon" | "sfadamw" | "ranger") {
            return Err("--backend cuda: --opt takes muon (default, Muon + Schedule-Free AdamW), sfadamw (--experiment only), or ranger (--experiment only)".into());
        }
        if exp.is_none() && opt_name != "muon" {
            return Err(format!(
                "--opt {opt_name} is forbidden on standard CUDA training (LAWS D5/D1260/D1629: mandatory default is --opt muon [Muon + Schedule-Free AdamW]); pass --experiment NAME under experiments/"
            ));
        }
        let legacy: Vec<&str> = ["--patience", "--plateau", "--max-drops", "--lr-min", "--steps", "--min-delta"]
            .into_iter()
            .filter(|k| f(k).is_some())
            .collect();
        if exp.is_none() && !legacy.is_empty() {
            return Err(format!(
                "{legacy:?} refused (LAWS C15/D5, user D1553): the run stops after 4 epochs of data without a val gain >= 1e-5 and the LR comes from the optimiser; another schedule needs --experiment NAME with --out under experiments/"
            ));
        }
        if let Some(n) = &exp {
            let under = f("--out").unwrap_or_default().replace('\\', "/").split('/').any(|c| c.eq_ignore_ascii_case("experiments"));
            if !under {
                return Err(format!("--experiment {n}: --out must be under a folder named 'experiments'"));
            }
        }
        #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
        let epoch_rule = exp.is_none();
        #[cfg(feature = "gpu")]
        {
            let g = |n: &str, d: u64| crate::flag_u64(args, n).unwrap_or(d) as usize;
            return crate::gpu::run(crate::gpu::GpuCfg {
                data: f("--data").ok_or("--data required")?,
                val: f("--val").ok_or("--val required")?,
                out: f("--out").ok_or("--out required")?,
                init: f("--init").unwrap_or_else(|| "bundled".into()),
                bs: g("--bs", 16384),
                steps: if epoch_rule { 1usize << 40 } else { g("--steps", 1000) },
                lr: crate::flag_f64(args, "--lr").unwrap_or(1e-3) as f32,
                lr_min: if epoch_rule {
                    crate::flag_f64(args, "--lr").unwrap_or(1e-3) as f32
                } else {
                    crate::flag_f64(args, "--lr-min").unwrap_or(2e-5) as f32
                },
                warmup: g("--warmup", 200),
                eval_every: g("--eval-every", 500),
                val_rows: g("--val-rows", 65536),
                threads: g("--threads", 8),
                seed: g("--seed", 7) as u64,
                d4,
                d4_all,
                holdout,
                resume: crate::has_flag(args, "--resume"),
                ram: crate::has_flag(args, "--ram"),
                wsum_clip: crate::flag_f64(args, "--wsum-clip").unwrap_or(0.0) as f32,
                ema: crate::flag_f64(args, "--ema").unwrap_or(0.0) as f32,
                guard: crate::flag_f64(args, "--guard").unwrap_or(1.15) as f32,
                patience: if epoch_rule { 0 } else { g("--patience", 0) },
                min_delta: if epoch_rule { 1e-5 } else { crate::flag_f64(args, "--min-delta").unwrap_or(2e-6) },
                plateau: if epoch_rule { 0 } else { g("--plateau", 0) },
                lr_factor: crate::flag_f64(args, "--lr-factor").unwrap_or(0.5) as f32,
                max_drops: if epoch_rule { 0 } else { g("--max-drops", 3) },
                epoch_rule,
                proof_blend: crate::flag_f64(args, "--proof-blend").unwrap_or(1.0) as f32,
                result_weight: targets::parse_rw(f("--result-weight-phase"), f("--result-weight-buckets"))?,
                loss_weight: targets::parse_lw(f("--loss-weight-buckets"))?,
                keep_all: crate::has_flag(args, "--keep-all"),
                sf: opt_name == "sfadamw",
                muon: if opt_name == "muon" {
                    Some((crate::flag_f64(args, "--muon-lr").unwrap_or(1e-3) as f32, crate::flag_f64(args, "--muon-mom").unwrap_or(0.95) as f32))
                } else {
                    None
                },
            });
        }
        #[cfg(not(feature = "gpu"))]
        return Err("this ti was built without --features cuda".into());
    }
    // engine D427 / D1629: standard CPU trainer enforces Muon + Schedule-Free AdamW + LAWS C15 unless --experiment NAME under experiments/
    let result_weight: Option<targets::Rw> = targets::parse_rw(f("--result-weight-phase"), f("--result-weight-buckets"))?;
    let loss_weight: Option<targets::Rw> = targets::parse_lw(f("--loss-weight-buckets"))?;
    let data = f("--data").ok_or("--data <stem> required (ti data pack output)")?;
    let val_stem = f("--val").ok_or("--val <stem> required")?;
    let out = f("--out").ok_or("--out <dir> required")?;
    let opt_kind = f("--opt").unwrap_or_else(|| "muonadamw".into());
    if exp.is_none() {
        let legacy: Vec<&str> = ["--steps", "--lr-min", "--patience"].into_iter().filter(|k| f(k).is_some()).collect();
        if !legacy.is_empty() {
            return Err(format!("{legacy:?} refused on standard CPU run (LAWS C15/D5); use --experiment NAME under experiments/"));
        }
        if !matches!(opt_kind.as_str(), "muonadamw" | "muon" | "muonsfadamw") {
            return Err(format!(
                "--opt {opt_kind} is forbidden on the standard path (LAWS D5/D1260/D1629: mandatory default is muonadamw/muon [Muon + Schedule-Free AdamW]); pass --experiment NAME under experiments/"
            ));
        }
    } else if !out.replace('\\', "/").split('/').any(|c| c.eq_ignore_ascii_case("experiments")) {
        return Err(format!("--experiment {}: --out must be under a folder named 'experiments'", exp.as_deref().unwrap_or("")));
    }
    let init = f("--init").unwrap_or_else(|| "bundled".into());
    let bs = crate::flag_u64(args, "--bs").unwrap_or(16384) as usize;
    let lr = crate::flag_f64(args, "--lr").unwrap_or(1e-3) as f32;
    let epoch_rule = exp.is_none();
    let steps = if epoch_rule { usize::MAX } else { crate::flag_u64(args, "--steps").unwrap_or(1000) as usize };
    let lr_min = if epoch_rule { lr } else { crate::flag_f64(args, "--lr-min").unwrap_or(2e-5) as f32 };
    let warmup = crate::flag_u64(args, "--warmup").unwrap_or(200) as usize;
    let wd = crate::flag_f64(args, "--wd").unwrap_or(0.0) as f32;
    let threads = crate::flag_u64(args, "--threads").unwrap_or(4).max(1) as usize;
    let val_rows = crate::flag_u64(args, "--val-rows").unwrap_or(65536) as usize;
    let save_every = crate::flag_u64(args, "--save-every").unwrap_or(500) as usize;
    let seed = crate::flag_u64(args, "--seed").unwrap_or(7);
    let resume = crate::has_flag(args, "--resume");
    let wsum_clip = crate::flag_f64(args, "--wsum-clip").unwrap_or(0.0) as f32;
    let guard = crate::flag_f64(args, "--guard").unwrap_or(1.15);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;

    let bytes = if init == "bundled" { BUNDLED_NNUE.to_vec() } else { std::fs::read(&init).map_err(|e| format!("{init}: {e}"))? };
    let mut net = TrainableNet::from_tnn_bytes(&bytes)?;
    let mut step0 = 0usize;
    let state = format!("{out}/state.txt");
    if resume {
        if let Ok(t) = std::fs::read_to_string(&state) {
            let mut it = t.split_whitespace();
            let (s, p) = (it.next().and_then(|x| x.parse().ok()).unwrap_or(0), it.next().unwrap_or("").to_string());
            net = TrainableNet::from_tnn_bytes(&std::fs::read(&p).map_err(|e| format!("{p}: {e}"))?)?;
            step0 = s;
            println!("[resume] step {step0} from {p} (optimizer moments restart)");
        }
    }
    let mut train = if crate::has_flag(args, "--ram") { Tdat::open_ram(&data)? } else { Tdat::open(&data)? };
    let d4_mult = if d4_all { mapper::MODES } else { 1 };
    let epoch_batches = (train.rows * d4_mult).div_ceil(bs).max(1);
    let eval_every = if epoch_rule { epoch_batches } else { crate::flag_u64(args, "--eval-every").unwrap_or(250) as usize };
    let mut valf = Tdat::open(&val_stem)?;
    let val = load_val(&mut valf, val_rows)?;
    println!("[train] rows {} val {} bs {bs} {} lr {lr}->{lr_min} warmup {warmup} threads {threads} d4 {d4} d4_all {d4_all}", train.rows, val.len(), if epoch_rule { format!("epoch_batches {epoch_batches} ({d4_mult}x D4, LAWS C15)") } else { format!("steps {steps}") });
    let v0 = evaluate(&net, &val, threads);
    if !std::path::Path::new(&format!("{out}/best.tnn")).exists() {
        // a best snapshot exists from the first second on (the init), however short the run (LAWS A14)
        crate::stopctl::write_atomic(&format!("{out}/best.tnn"), &net.to_tnn_bytes())?;
    }
    if let Some(f) = crate::stopctl::requested(Some(std::path::Path::new(&out))) {
        return Err(format!("stop file {f} exists: remove it to train"));
    }
    println!("[val step {step0}] mse {:.6} proven {:.6} (n={}) other {:.6}   {}", v0.mse, v0.mse_proven, v0.n_proven, v0.mse_other, tensor_report(&net));
    if let Some(lw) = loss_weight {
        println!("[loss-weight] {lw:?} by NNUE bucket (16)");
    }

    let mut opt = SparseAdamW::new(&net);
    // D1159 / D1162 / D1629: Muon + Schedule-Free AdamW is the mandatory standard CPU optimiser
    let muon_lr = crate::flag_f64(args, "--muon-lr").unwrap_or(1e-3) as f32;
    match opt_kind.as_str() {
        "adamw" | "sfadamw" => {
            opt.muon = None;
            println!("[train] EXPERIMENT optimiser: Schedule-Free AdamW ONLY (NO MUON)");
        }
        "muonadamw" | "muon" | "muonsfadamw" => {
            let mom = crate::flag_f64(args, "--muon-mom").unwrap_or(0.95) as f32;
            opt.muon = Some(crate::train::Muon::new(&net, muon_lr, mom));
            println!("[train] optimiser: Muon + Schedule-Free AdamW (Muon lr {muon_lr}, momentum {mom}, on l1/l2; Schedule-Free AdamW on the rest; eval/export on x)");
        }
        k => return Err(format!("--opt {k}: use muonadamw/muon (default, Muon + Schedule-Free AdamW) or sfadamw/adamw (--experiment)")),
    }
    let mut grads: Vec<BatchGradients> = (0..threads).map(|_| BatchGradients::new(&net)).collect();
    let mut main_grad = BatchGradients::new(&net);
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for _ in 0..step0 {
        rng.next();
    }
    let nblk = train.rows / BLOCK;
    let mut raw: Vec<Rec> = Vec::new();
    let mut buf: Vec<Rec> = Vec::new();
    let mut perm = if d4_all {
        let mut nx = || rng.next();
        let mut p = mapper::BlockPerm::new(nblk, &mut nx);
        if nblk > 0 {
            p.advance((step0 * bs / mapper::MODES / BLOCK) % nblk, &mut nx);
        }
        println!("[train] d4-all FULL 8x SCRAMBLED D4 QUEUE (default): all 8 orientations per board per pass");
        Some(p)
    } else {
        None
    };
    let mut best = v0.mse;
    let mut best_saved = v0.mse;
    let mut flat_epochs = 0usize;
    let t0 = Instant::now();
    let (mut run_loss, mut run_w, mut run_n) = (0f64, 0f64, 0usize);
    for step in step0..steps {
        if buf.len() < bs {
            if let Some(p) = perm.as_mut() {
                raw.clear();
                for _ in 0..(bs * 8 / BLOCK).max(16) {
                    let b = {
                        let mut nx = || rng.next();
                        p.next_block(&mut nx)
                    };
                    train.read_block(b * BLOCK, BLOCK, &mut raw)?;
                }
                mapper::expand_and_scramble_d4(&raw, &mut buf, || rng.next());
            } else {
                // refill: 64 batches worth of random 1024-row blocks, shuffled together (mixes sources and phases)
                buf.clear();
                for _ in 0..(bs * 64 / BLOCK).max(1) {
                    train.read_block(rng.below(nblk) * BLOCK, BLOCK, &mut buf)?;
                }
                for i in (1..buf.len()).rev() {
                    let j = rng.below(i + 1);
                    buf.swap(i, j);
                }
            }
        }
        let batch: Vec<Rec> = buf.split_off(buf.len() - bs);
        let modes: Vec<usize> = batch.iter().map(|_| if d4 && !d4_all { rng.below(8) } else { 0 }).collect();
        let per = (bs + threads - 1) / threads;
        let netref = &net;
        let losses: Vec<(f64, f64)> = std::thread::scope(|s| {
            grads
                .iter_mut()
                .zip(batch.chunks(per).zip(modes.chunks(per)))
                .map(|(g, (rs, ms))| {
                    s.spawn(move || {
                        let (mut l, mut wsum) = (0f64, 0f64);
                        for (r, &m) in rs.iter().zip(ms) {
                            let t = targets::target_train_rw(r, 1.0, result_weight);
                            let mut w = if wsum_clip > 0.0 { r.wsum.clamp(0.0, wsum_clip) } else { r.wsum.max(0.0) };
                            if let Some(lw) = loss_weight {
                                w *= lw[targets::rw_bucket(r)];
                            }
                            if w == 0.0 {
                                continue;
                            }
                            l += netref.forward_backward_weighted(&mapper::board_of(r, m), t.target, w, g) as f64;
                            wsum += w as f64;
                        }
                        (l, wsum)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        main_grad.clear_sparse(net.width, net.buckets);
        for g in grads.iter_mut() {
            main_grad.merge_from(g, net.width, net.buckets);
            g.clear_sparse(net.width, net.buckets);
        }
        let wsum: f64 = losses.iter().map(|x| x.1).sum();
        let cur_lr = lr_at(step, steps, lr, lr_min, warmup);
        if let Some(m) = opt.muon.as_mut() {
            m.lr = muon_lr * cur_lr / lr;
        }
        opt.step_batch(&mut net, &main_grad, wsum.max(1.0) as usize, cur_lr, wd);
        run_loss += losses.iter().map(|x| x.0).sum::<f64>();
        run_w += wsum;
        run_n += 1;
        let done = step + 1;
        // LAWS A14 (user D1307): STOP in --out or C:/gitProjects/STOP_ALL: save the exact step as a checkpoint and exit
        if let Some(f) = crate::stopctl::requested(Some(std::path::Path::new(&out))) {
            opt.sync_eval(&mut net);
            let p = format!("{out}/ckpt_{done}.tnn");
            crate::stopctl::write_atomic(&p, &net.to_tnn_bytes())?;
            crate::stopctl::write_atomic(&state, format!("{done} {p}").as_bytes())?;
            println!("[train] STOPPED by {f} after step {done}: checkpoint {p}, best {out}/best.tnn; the same command + --resume continues (optimizer moments restart)");
            return Ok(());
        }
        if done % 50 == 0 {
            println!("[step {done}/{steps}] loss {:.6} lr {:.2e} {:.0} rows/s", run_loss / run_w.max(1.0), lr_at(step, steps, lr, lr_min, warmup), (done - step0) as f64 * bs as f64 / t0.elapsed().as_secs_f64());
            run_loss = 0.0;
            run_w = 0.0;
            run_n = 0;
        }
        let need_eval = done % eval_every == 0 || done == steps;
        let need_save = done % save_every == 0 || done == steps;
        if need_eval || need_save {
            opt.sync_eval(&mut net);
        }
        let mut saturated = false;
        if need_eval {
            let v = evaluate(&net, &val, threads);
            println!("[val step {done}] mse {:.6} proven {:.6} other {:.6}", v.mse, v.mse_proven, v.mse_other);
            if guard > 0.0 && v.mse > v0.mse * guard {
                return Err(format!("GUARD TRIPPED at step {done}: val mse {:.6} > {guard} x init {:.6} (Mistake 39)", v.mse, v0.mse));
            }
            if v.mse < best_saved {
                best_saved = v.mse;
                let b = net.to_tnn_bytes();
                crate::stopctl::write_atomic(&format!("{out}/best.tnn"), &b)?;
                crate::prov::record(&out, "best.tnn", &b, v.mse, &val_stem, val_rows, done);
            }
            if v.mse <= best - 1e-5 {
                best = v.mse;
                flat_epochs = 0;
            } else if epoch_rule {
                flat_epochs += 1;
                if flat_epochs >= 4 {
                    println!("[epoch rule] SATURATED (LAWS C15, D1553): 4 epochs without a val gain >= 1e-5 (best {best:.6}, saved {best_saved:.6})");
                    saturated = true;
                }
            }
        }
        if need_save {
            let p = format!("{out}/ckpt_{done}.tnn");
            crate::stopctl::write_atomic(&p, &net.to_tnn_bytes())?;
            crate::stopctl::write_atomic(&state, format!("{done} {p}").as_bytes())?;
        }
        if saturated {
            break;
        }
        if need_eval || need_save {
            opt.sync_train(&mut net);
        }
    }
    let _ = run_n;
    opt.sync_eval(&mut net);
    std::fs::write(format!("{out}/final.tnn"), net.to_tnn_bytes()).map_err(|e| e.to_string())?;
    println!("[done] {} | {}", tensor_report(&net), targets::MATE);
    Ok(())
}

/// `ti train parity --net X|bundled --tdat stem [--n 20000]`: hub f32 forward vs engine f32 forward (must agree ~1e-4)
/// and vs the engine's quantized eval (reported, cp units).
pub fn run_parity(args: &[String]) -> ExitCode {
    let net_spec = crate::flag_str(args, "--net").unwrap_or_else(|| "bundled".into());
    let Some(stem) = crate::flag_str(args, "--tdat") else {
        eprintln!("usage: ti train parity --net <x.tnn|bundled> --tdat <stem> [--n 20000]");
        return ExitCode::FAILURE;
    };
    let n = crate::flag_u64(args, "--n").unwrap_or(20000) as usize;
    let bytes = if net_spec == "bundled" { BUNDLED_NNUE.to_vec() } else { std::fs::read(&net_spec).unwrap() };
    let net = TrainableNet::from_tnn_bytes(&bytes).unwrap();
    let eng = if net_spec == "bundled" { Nnue::bundled() } else { Nnue::load(&net_spec).unwrap() };
    let mut d = Tdat::open(&stem).unwrap();
    let rows = load_val(&mut d, n).unwrap();
    let (mut worst_f, mut worst_q, mut sum_q) = (0f32, 0f32, 0f64);
    for r in &rows {
        let b = mapper::board_of(r, 0);
        let y = net.forward_y(&b);
        worst_f = worst_f.max((y - eng.forward_f32(&b, 0.0)).abs());
        let dq = (400.0 * y - eng.eval(&b, 0) as f32).abs();
        worst_q = worst_q.max(dq);
        sum_q += dq as f64;
    }
    println!("rows {}  max|hub y - engine f32 y| = {:.2e}   quantized-vs-f32 cp: max {:.2} mean {:.3}", rows.len(), worst_f, worst_q, sum_q / rows.len() as f64);
    if worst_f < 1e-3 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

/// `ti train check --data stem --val stem [--init net]`: preflight. Refuses (exit 1) on any red; prints every check.
pub fn run_check(args: &[String]) -> ExitCode {
    let f = |n: &str| crate::flag_str(args, n);
    let (Some(data), Some(val)) = (f("--data"), f("--val")) else {
        eprintln!("usage: ti train check --data <stem> --val <stem> [--init net.tnn|bundled]");
        return ExitCode::FAILURE;
    };
    let init = f("--init").unwrap_or_else(|| "bundled".into());
    let mut red = 0;
    let mut chk = |ok: bool, what: String| {
        println!("[{}] {what}", if ok { "ok " } else { "RED" });
        if !ok {
            red += 1;
        }
    };
    let (mut td, mut vd) = match (Tdat::open(&data), Tdat::open(&val)) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            eprintln!("open failed: {:?} {:?}", a.err(), b.err());
            return ExitCode::FAILURE;
        }
    };
    for (name, stem, rows) in [("train", &data, td.rows), ("val", &val, vd.rows)] {
        match std::fs::read_to_string(format!("{stem}.manifest.json")) {
            Ok(m) => chk(m.contains(&format!("\"rows\":{rows},")), format!("{name}: manifest rows == tdat rows ({rows})")),
            Err(_) => chk(false, format!("{name}: manifest missing")),
        }
    }
    let bytes = if init == "bundled" { Some(BUNDLED_NNUE.to_vec()) } else { std::fs::read(&init).ok() };
    let net = bytes.as_deref().and_then(|b| TrainableNet::from_tnn_bytes(b).ok());
    chk(net.is_some(), format!("init net parses ({init})"));
    if let Some(net) = &net {
        chk(net.has_dis && net.buckets == 8 && net.width > 0, format!("TNN3 W={} buckets={} dis head {}", net.width, net.buckets, net.has_dis));
        let (mut rows, mut other) = (Vec::new(), Vec::new());
        let nblk = td.rows / BLOCK;
        for i in 0..16 {
            td.read_block((i * nblk / 16) * BLOCK, BLOCK, &mut rows).unwrap();
        }
        vd.read_block(0, BLOCK, &mut other).unwrap();
        let bad = rows.iter().chain(&other).filter(|r| !(r.cp.is_finite() && r.wsum.is_finite() && r.wsum >= 0.0 && (0.0..=1.0).contains(&r.outcome) && r.stm < 2 && r.occ0 & r.occ1 == 0 && (r.occ0 | r.occ1) >> 49 == 0)).count();
        chk(bad == 0, format!("{} sampled rows sane (finite cp/wsum, outcome in [0,1], disjoint 49-bit boards): {bad} bad", rows.len() + other.len()));
        let proven = rows.iter().filter(|r| target_of(r).proven >= 0).count();
        chk(proven > 0, format!("proven rows seen in train sample: {proven}/{}", rows.len()));
        let mut worst = 0f32;
        let eng = if init == "bundled" { Nnue::bundled() } else { Nnue::load(&init).unwrap() };
        for r in rows.iter().take(2000) {
            let b = mapper::board_of(r, 0);
            worst = worst.max((net.forward_y(&b) - eng.forward_f32(&b, 0.0)).abs());
        }
        chk(worst < 1e-3, format!("hub forward == engine f32 forward on 2000 rows (max diff {worst:.1e})"));
        let ma = |v: &Vec<f32>| v.iter().fold(0f32, |a, &x| a.max(x.abs()));
        chk(ma(&net.ft) <= 1.98 && ma(&net.ftb) <= 1.98, "ft/ftb inside the +-1.98 clip (int16 at QA 1024)".into());
    }
    println!("{}", if red == 0 { "PREFLIGHT GREEN" } else { "PREFLIGHT RED" });
    if red == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

/// `ti train buckets --net n.tnn --data <stem> [--rows 65536]`: where does the remaining loss live? MSE per phase bucket
/// (stones/empties), split proven (exact 1/0 targets) vs other, next to the variance of the target in that bucket
/// (mse / variance = share of the target the net does NOT explain; ~1 = no better than predicting the bucket mean).
pub fn run_buckets(args: &[String]) -> ExitCode {
    let f = |n: &str| crate::flag_str(args, n);
    let (Some(net_p), Some(data)) = (f("--net"), f("--data")) else {
        eprintln!("usage: ti train buckets --net <n.tnn> --data <stem> [--rows 65536] [--by-empties] [--dump rows.txt]");
        return ExitCode::FAILURE;
    };
    let rows = crate::flag_u64(args, "--rows").unwrap_or(65536) as usize;
    let net = match std::fs::read(&net_p).map_err(|e| e.to_string()).and_then(|b| TrainableNet::from_tnn_bytes(&b)) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("buckets: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut t = match Tdat::open(&data) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("buckets: {e}");
            return ExitCode::FAILURE;
        }
    };
    let v = match load_val(&mut t, rows) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("buckets: {e}");
            return ExitCode::FAILURE;
        }
    };
    // --by-empties: one row per empty-square count (0..=49) instead of per net bucket (to design a bucket map)
    let by_empties = crate::has_flag(args, "--by-empties");
    let nb = if by_empties { 50 } else { net.buckets };
    // per bucket: n, sum t, sum t^2, sum err^2, n_proven, err^2 proven
    let mut a = vec![[0f64; 6]; nb];
    // per-empties sufficient statistics of the OLS fit target ~ a + b * p: n, sum p, sum t, sum p^2, sum p*t, sum t^2
    let mut st = vec![[0f64; 6]; 50];
    // --dump FILE: one line per row "row empties bucket p target proven", for paired comparisons of nets on the same rows
    let mut dump = crate::flag_str(args, "--dump").map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("buckets: --dump")));
    for (i, r) in v.iter().enumerate() {
        let tg = target_of(r);
        let fw = net.forward_full(&mapper::board_of(r, 0));
        let p = 1.0 / (1.0 + (-fw.y as f64).exp());
        let e = (p - tg.target as f64).powi(2);
        if let Some(w) = dump.as_mut() {
            use std::io::Write;
            let em = mapper::board_of(r, 0).empty().count_ones();
            writeln!(w, "{i} {em} {} {p:.6} {:.6} {}", fw.bucket, tg.target, tg.proven).expect("buckets: --dump write");
        }
        if by_empties {
            let e = (mapper::board_of(r, 0).empty().count_ones() as usize).min(49);
            let (pp, tt) = (p, tg.target as f64);
            let q = &mut st[e];
            q[0] += 1.0;
            q[1] += pp;
            q[2] += tt;
            q[3] += pp * pp;
            q[4] += pp * tt;
            q[5] += tt * tt;
        }
        let x = &mut a[if by_empties { mapper::board_of(r, 0).empty().count_ones() as usize } else { fw.bucket }];
        x[0] += 1.0;
        x[1] += tg.target as f64;
        x[2] += (tg.target as f64).powi(2);
        x[3] += e;
        if tg.proven >= 0 {
            x[4] += 1.0;
            x[5] += e;
        }
    }
    println!("{} rows of {data}, net {net_p}", v.len());
    println!("bucket   rows   mse     target-var  unexplained  proven-rows  proven-mse  share-of-total-loss");
    let tot: f64 = a.iter().map(|x| x[3]).sum();
    for (b, x) in a.iter().enumerate() {
        if x[0] == 0.0 {
            continue;
        }
        let mean = x[1] / x[0];
        let var = x[2] / x[0] - mean * mean;
        let mse = x[3] / x[0];
        println!("{b:>6} {:>6} {:.5}  {:.5}     {:>6.1}%      {:>6}      {:.5}     {:>5.1}%", x[0] as usize, mse, var, 100.0 * mse / var.max(1e-12), x[4] as usize, if x[4] > 0.0 { x[5] / x[4] } else { 0.0 }, 100.0 * x[3] / tot.max(1e-12));
    }
    if by_empties {
        bucket_plan(&st, crate::flag_u64(args, "--plan").map(|x| x as usize));
    }
    ExitCode::SUCCESS
}

/// Where should the bucket boundaries go? For a fixed trunk, one bucket = one (a, b) recalibration of the output
/// (target ~ a + b * p). Cost of a run of empties counts = squared error left after the OLS fit of that run; the DP picks the
/// K contiguous runs over empties 1..=45 with the least total error. Compared with the even and E16 maps and with the net as it is.
fn bucket_plan(st: &[[f64; 6]], only_k: Option<usize>) {
    let lo = 1usize;
    let hi = 45usize;
    let total: f64 = (lo..=hi).map(|e| st[e][0]).sum();
    let cost_run = |i: usize, j: usize| -> f64 {
        let mut q = [0f64; 6];
        for e in i..=j {
            for k in 0..6 {
                q[k] += st[e][k];
            }
        }
        let n = q[0];
        if n < 2.0 {
            return 0.0;
        }
        let (stt, spp, spt) = (q[5] - q[2] * q[2] / n, q[3] - q[1] * q[1] / n, q[4] - q[1] * q[2] / n);
        if spp < 1e-9 {
            return stt;
        }
        stt - spt * spt / spp
    };
    let raw: f64 = (lo..=hi).map(|e| {
        let q = &st[e];
        // squared error of p itself: sum (p - t)^2 = pp - 2pt + tt
        q[3] - 2.0 * q[4] + q[5]
    }).sum();
    println!("
--- bucket plan over empties {lo}..={hi} ({total:.0} rows): mse of the net as is {:.5}", raw / total);
    let n = hi - lo + 1;
    let maxk = 40usize.min(n);
    let mut c = vec![vec![0f64; n]; n]; // c[i][j]: run lo+i ..= lo+j
    for i in 0..n {
        for j in i..n {
            c[i][j] = cost_run(lo + i, lo + j);
        }
    }
    let mut f = vec![vec![f64::INFINITY; n]; maxk + 1];
    let mut cut = vec![vec![0usize; n]; maxk + 1];
    for j in 0..n {
        f[1][j] = c[0][j];
    }
    for k in 2..=maxk {
        for j in (k - 1)..n {
            for i in (k - 1)..=j {
                let v = f[k - 1][i - 1] + c[i][j];
                if v < f[k][j] {
                    f[k][j] = v;
                    cut[k][j] = i;
                }
            }
        }
    }
    let eval_map = |starts: &[usize]| -> f64 {
        let mut s = 0.0;
        for (b, &st0) in starts.iter().enumerate() {
            let en = if b + 1 < starts.len() { starts[b + 1] - 1 } else { hi };
            let (i, j) = (st0.max(lo), en.min(hi));
            if i <= j {
                s += cost_run(i, j);
            }
        }
        s
    };
    let even = |k: usize| -> Vec<usize> {
        let mut v = vec![0usize];
        for b in 1..k {
            v.push((0..=49).find(|&e| mapper::bucket_of(e, k) == b).unwrap_or(50));
        }
        v
    };
    println!("  K   best-mse   even-mse   gain-vs-even   starts of the best map");
    for k in [4usize, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40] {
        if k > maxk || only_k.map_or(false, |o| o != k) {
            continue;
        }
        let mut starts = Vec::new();
        let mut j = n - 1;
        for kk in (1..=k).rev() {
            let i = if kk == 1 { 0 } else { cut[kk][j] };
            starts.push(lo + i);
            if kk > 1 {
                j = i - 1;
            }
        }
        starts.reverse();
        let best = f[k][n - 1] / total;
        let ev = eval_map(&even(k)) / total;
        println!("{k:>3}   {best:.5}    {ev:.5}    {:>+7.1}%      {:?}", 100.0 * (best - ev) / ev, starts);
    }
    println!("  E16 map (current plan) mse {:.5}", eval_map(&mapper::E16_STARTS) / total);
}

#[cfg(test)]
mod tests {
    use super::run_train_inner;

    fn args(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn refuses_suboptimal_opt_without_experiment() {
        for bad in [
            vec!["--backend", "cuda", "--opt", "sfadamw", "--data", "d", "--val", "v", "--out", "o"],
            vec!["--backend", "cuda", "--opt", "ranger", "--data", "d", "--val", "v", "--out", "o"],
            vec!["--opt", "adamw", "--data", "d", "--val", "v", "--out", "o"],
            vec!["--opt", "sfadamw", "--data", "d", "--val", "v", "--out", "o"],
        ] {
            let err = run_train_inner(&args(&bad)).expect_err("suboptimal --opt must be refused");
            assert!(err.contains("--experiment"), "expected --experiment hint in: {err}");
        }
        for bad in [
            vec!["--d4-all", "--no-d4", "--data", "d", "--val", "v", "--out", "o"],
            vec!["--d4-all", "--d4-rand", "--data", "d", "--val", "v", "--out", "o"],
            vec!["--d4-all", "--holdout-mode", "0", "--data", "d", "--val", "v", "--out", "o"],
            vec!["--d4-rand", "--data", "d", "--val", "v", "--out", "o"],
            vec!["--no-d4", "--data", "d", "--val", "v", "--out", "o"],
        ] {
            let err = run_train_inner(&args(&bad)).expect_err("non-default D4 flags must be refused");
            assert!(err.contains("--d4-all") || err.contains("--experiment"), "unexpected error: {err}");
        }
    }
}
