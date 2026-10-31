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

pub struct Tdat {
    f: File,
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
        Ok(Tdat { f, rows: len / REC })
    }
    pub fn read_block(&mut self, start: usize, n: usize, out: &mut Vec<Rec>) -> Result<(), String> {
        let n = n.min(self.rows - start);
        self.f.seek(SeekFrom::Start((start * REC) as u64)).map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; n * REC];
        self.f.read_exact(&mut buf).map_err(|e| e.to_string())?;
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
    let part = (val.len() + threads - 1) / threads;
    let res: Vec<(f64, f64, usize, f64)> = std::thread::scope(|s| {
        val.chunks(part.max(1))
            .map(|ch| {
                s.spawn(move || {
                    let (mut a, mut p, mut np, mut o) = (0f64, 0f64, 0usize, 0f64);
                    for r in ch {
                        let t = target_of(r);
                        let y = net.forward_y(&mapper::board_of(r, 0));
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
    let data = f("--data").ok_or("--data <stem> required (ti data pack output)")?;
    let val_stem = f("--val").ok_or("--val <stem> required")?;
    let out = f("--out").ok_or("--out <dir> required")?;
    let init = f("--init").unwrap_or_else(|| "bundled".into());
    let bs = crate::flag_u64(args, "--bs").unwrap_or(16384) as usize;
    let steps = crate::flag_u64(args, "--steps").unwrap_or(1000) as usize;
    let lr = crate::flag_f64(args, "--lr").unwrap_or(1e-3) as f32;
    let lr_min = crate::flag_f64(args, "--lr-min").unwrap_or(2e-5) as f32;
    let warmup = crate::flag_u64(args, "--warmup").unwrap_or(200) as usize;
    let wd = crate::flag_f64(args, "--wd").unwrap_or(0.0) as f32;
    let threads = crate::flag_u64(args, "--threads").unwrap_or(4).max(1) as usize;
    let val_rows = crate::flag_u64(args, "--val-rows").unwrap_or(65536) as usize;
    let eval_every = crate::flag_u64(args, "--eval-every").unwrap_or(250) as usize;
    let save_every = crate::flag_u64(args, "--save-every").unwrap_or(500) as usize;
    let seed = crate::flag_u64(args, "--seed").unwrap_or(7);
    let d4 = !crate::has_flag(args, "--no-d4");
    let resume = crate::has_flag(args, "--resume");
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
    let mut train = Tdat::open(&data)?;
    let mut valf = Tdat::open(&val_stem)?;
    let val = load_val(&mut valf, val_rows)?;
    println!("[train] rows {} val {} bs {bs} steps {steps} lr {lr}->{lr_min} warmup {warmup} threads {threads} d4 {d4}", train.rows, val.len());
    let v0 = evaluate(&net, &val, threads);
    println!("[val step {step0}] mse {:.6} proven {:.6} (n={}) other {:.6}   {}", v0.mse, v0.mse_proven, v0.n_proven, v0.mse_other, tensor_report(&net));

    let mut opt = SparseAdamW::new(&net);
    let mut grads: Vec<BatchGradients> = (0..threads).map(|_| BatchGradients::new(&net)).collect();
    let mut main_grad = BatchGradients::new(&net);
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for _ in 0..step0 {
        rng.next();
    }
    let nblk = train.rows / BLOCK;
    let mut buf: Vec<Rec> = Vec::new();
    let mut best = v0.mse;
    let t0 = Instant::now();
    let (mut run_loss, mut run_w, mut run_n) = (0f64, 0f64, 0usize);
    for step in step0..steps {
        if buf.len() < bs {
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
        let batch: Vec<Rec> = buf.split_off(buf.len() - bs);
        let modes: Vec<usize> = batch.iter().map(|_| if d4 { rng.below(8) } else { 0 }).collect();
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
                            let t = target_of(r);
                            let w = r.wsum.max(0.0);
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
        opt.step_batch(&mut net, &main_grad, wsum.max(1.0) as usize, lr_at(step, steps, lr, lr_min, warmup), wd);
        run_loss += losses.iter().map(|x| x.0).sum::<f64>();
        run_w += wsum;
        run_n += 1;
        let done = step + 1;
        if done % 50 == 0 {
            println!("[step {done}/{steps}] loss {:.6} lr {:.2e} {:.0} rows/s", run_loss / run_w.max(1.0), lr_at(step, steps, lr, lr_min, warmup), (done - step0) as f64 * bs as f64 / t0.elapsed().as_secs_f64());
            run_loss = 0.0;
            run_w = 0.0;
            run_n = 0;
        }
        if done % eval_every == 0 || done == steps {
            let v = evaluate(&net, &val, threads);
            println!("[val step {done}] mse {:.6} proven {:.6} other {:.6}", v.mse, v.mse_proven, v.mse_other);
            if v.mse < best {
                best = v.mse;
                std::fs::write(format!("{out}/best.tnn"), net.to_tnn_bytes()).map_err(|e| e.to_string())?;
            }
        }
        if done % save_every == 0 || done == steps {
            let p = format!("{out}/ckpt_{done}.tnn");
            std::fs::write(&p, net.to_tnn_bytes()).map_err(|e| e.to_string())?;
            std::fs::write(&state, format!("{done} {p}")).map_err(|e| e.to_string())?;
        }
    }
    let _ = run_n;
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
