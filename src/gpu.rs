//! H8: GPU backend. The same net as `TrainableNet`, expressed as a bullet graph (pinned rev, see Cargo.toml), fed by
//! the hub's own reader / D4 mapper / targets, optimised with Ranger, validated by the hub CPU `evaluate`.
//! Built only with `--features cuda` (needs the CUDA toolkit; Modal image). `ti train gpu-parity` proves the graph
//! reproduces the hub forward before any training step.
use std::sync::Arc;

use bullet_compiler::tensor::TValue;
use bullet_gpu::runtime::{Device, Gpu};
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights},
    optimiser::{ranger::{Ranger, RangerParams}, Optimiser},
    reader::{DataReader, ReadMapLoader},
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
};
use titanium::nnfeat::{MAX_ACTIVE, NUM_FEATS};
use titanium::nnue::BUNDLED_NNUE;

use crate::data::Rec;
use crate::layout::{self, GraphTensors};
use crate::mapper;
use crate::targets::{rw_val_row, target_train_rw};
use crate::train::TrainableNet;
use crate::trainer::{self, Tdat};

const BLOCK: usize = 1024;

/// Endless random-block sampler over a `.tdat` (blocks of 1024 rows, shuffled together in 128k-row windows).
#[derive(Clone)]
pub struct TdatReader {
    path: String,
    seed: u64,
    /// shared RAM copy (`--ram`), None = stream from disk
    ram: Option<std::sync::Arc<Tdat>>,
}

impl DataReader<Rec> for TdatReader {
    fn read_chunks<F: FnMut(&[Rec]) -> bool>(&self, skip_count: usize, mut f: F) {
        let mut t = match &self.ram {
            Some(r) => r.share(&self.path).expect("share tdat"),
            None => Tdat::open(&self.path).expect("open tdat"),
        };
        let nblk = t.rows / BLOCK;
        let mut s = (self.seed ^ (skip_count as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut buf: Vec<Rec> = Vec::with_capacity(128 * BLOCK);
        loop {
            buf.clear();
            for _ in 0..128 {
                t.read_block((next() % nblk as u64) as usize * BLOCK, BLOCK, &mut buf).expect("read block");
            }
            for i in (1..buf.len()).rev() {
                buf.swap(i, (next() % (i as u64 + 1)) as usize);
            }
            if f(&buf) {
                return;
            }
        }
    }
}

pub struct GpuCfg {
    pub data: String,
    pub val: String,
    pub out: String,
    pub init: String,
    pub bs: usize,
    pub steps: usize,
    pub lr: f32,
    pub lr_min: f32,
    pub warmup: usize,
    pub eval_every: usize,
    pub val_rows: usize,
    pub threads: usize,
    pub seed: u64,
    pub d4: bool,
    /// DIAGNOSTIC ONLY: never train on D4 orientation k, validate it separately. A big gap = the net memorised orientations
    /// instead of learning the symmetry. Production runs train on all 8 (search meets every orientation).
    pub holdout: Option<usize>,
    pub resume: bool,
    pub ram: bool,
    /// cap on the per-row loss weight (0 = off); Opus review: max wsum is 37.7, mean 1.2
    pub wsum_clip: f32,
    /// EMA of the weights over superbatches (0 = off): ema = d*ema + (1-d)*live, validated next to the live weights
    pub ema: f32,
    /// abort when val mse > guard * initial val mse (0 = off). Mistake 39 / D153: a bad fine-tune showed 0.0204 -> 0.0216 in epoch 1
    pub guard: f32,
    /// early stopping: stop after this many checkpoints without a val-mse gain > min_delta (0 = off). User order 2026-10-31.
    pub patience: usize,
    pub min_delta: f64,
    /// reduce-on-plateau (0 = off): checkpoints without a val gain before lr *= lr_factor; stop after max_drops useless drops
    pub plateau: usize,
    pub lr_factor: f32,
    pub max_drops: usize,
    /// D1553/D1554 (LAWS C15 + D5): epochs counted from the train rows; 4 epochs in a row without a val gain >= min_delta
    /// stop the run (the LR itself comes from the optimiser, never a hand rule; patience / plateau / max_drops / steps unused)
    pub epoch_rule: bool,
    /// proven rows train on blend*proof + (1-blend)*sigmoid(cp5k); 1.0 = exact proofs (default, D162)
    pub proof_blend: f32,
    /// game-result weight per NNUE phase bucket (16; the 3-value `--result-weight-phase` is expanded, engine D310 / D1275); val rows follow it
    pub result_weight: Option<crate::targets::Rw>,
    /// D1282: per-NNUE-bucket sample loss multiplier (16 values)
    pub loss_weight: Option<crate::targets::Rw>,
    /// keep every checkpoint (ckpt_<step>.tnn) so a strength-vs-step curve can be gated afterwards
    pub keep_all: bool,
    /// D1181: Some((muon_lr, momentum)) = `--opt muon`: Muon on the l1w / l2w bucket blocks, Ranger on the rest; None = Ranger only (default)
    pub muon: Option<(f32, f32)>,
    /// D1554: `--opt sfadamw` = Schedule-Free AdamW on every tensor (validated / exported on the averaged iterate x)
    pub sf: bool,
}

/// Inputs, mapper and graph. Weight ids: ftw/ftb, psqw/psqb (psqb cancels in stm-ntm), l1w/l1b, l2w/l2b, outw/outb.
fn build(net: &TrainableNet, d4: bool, holdout: Option<usize>, wsum_clip: f32, proof_blend: f32, rw: Option<crate::targets::Rw>, lw: Option<crate::targets::Rw>) -> (ModelInputsMapper<Rec>, ModelDefinition) {
    let (w, l1, l2, oi) = (net.width, net.l1, net.l2, net.out_in());
    let nb = net.buckets;
    let map_id = net.groups >> 8; // bucket map id (copied: the sampler closure must be 'static)
    assert!(!net.pairwise && net.s57w.is_empty() && nb >= 1 && net.num_feats == NUM_FEATS,
        "graph covers non-pairwise nets without the S57 skip, any bucket count, 26196 features (TNN1/TNN3)");
    let skip = net.v2; // TNN2: the output reads h1 || h2; TNN1/TNN3: h2 only (out_in = l2)
    let inputs = ModelInputs::default()
        .add_sparse("stm", (NUM_FEATS, 1), MAX_ACTIVE)
        .add_sparse("ntm", (NUM_FEATS, 1), MAX_ACTIVE)
        .add_sparse("bucket", (nb, 1), 1)
        .add_dense("target", (1, 1))
        .add_dense("wt", (1, 1));

    let mapper = ModelInputsMapper::build(&inputs, move |r: &Rec, step, ((((stm, ntm), bk), target), wt)| {
        let mode = if d4 {
            let h = (r.occ0 ^ r.occ1.rotate_left(23)).wrapping_add(step.total_batches() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            match holdout {
                // 7 trained orientations: draw 0..6 and skip over the held-out one
                Some(k) => {
                    let m = ((h >> 40) % 7) as usize;
                    if m >= k { m + 1 } else { m }
                }
                None => ((h >> 40) & 7) as usize,
            }
        } else {
            0
        };
        let s = mapper::sample(r, mode, nb, map_id);
        for (dst, src) in [(&mut *stm, &s.us), (&mut *ntm, &s.them)] {
            for (i, &f) in src.iter().enumerate() {
                dst[i] = f as i32;
            }
            if src.len() < MAX_ACTIVE {
                dst[src.len()] = -1;
            }
        }
        bk[0] = s.bucket as i32;
        target[0] = target_train_rw(r, proof_blend, rw).target;
        let w = if wsum_clip > 0.0 { r.wsum.clamp(0.0, wsum_clip) } else { r.wsum.max(0.0) };
        wt[0] = match lw {
            Some(a) => w * a[crate::targets::rw_bucket(r)],
            None => w,
        };
    });

    let defn = ModelDefinition::build(&inputs, move |b, ((((stm, ntm), bucket), target), wt)| {
        let ft = b.new_affine("ft", NUM_FEATS, w);
        let ps = b.new_affine("psq", NUM_FEATS, nb);
        let a1 = b.new_affine("l1", 2 * w, nb * l1);
        let a2 = b.new_affine("l2", l1, nb * l2);
        let ao = b.new_affine("out", oi, nb);

        let x = ft.forward(stm).crelu().concat(ft.forward(ntm).crelu());
        let h1 = a1.forward(x).select(bucket).crelu();
        let h2 = a2.forward(h1).select(bucket).crelu();
        let o = if skip { ao.forward(h1.concat(h2)) } else { ao.forward(h2) }.select(bucket);
        let p = (ps.forward(stm) - ps.forward(ntm)).select(bucket);
        let y = o + 0.5f32 * p;
        let loss = (y.sigmoid().power_error(target, 2.5) * wt).reduce_sum_batch();
        (Some(loss), vec![("output".to_string(), y)])
    });
    (mapper, defn)
}

fn set(w: &mut ModelWeights, id: &str, v: Vec<f32>) {
    assert!(w.set(id, TValue::F32(v)), "weight {id} not in graph");
}

fn get(w: &ModelWeights, id: &str) -> Vec<f32> {
    match &w.get(id).values {
        TValue::F32(v) => v.clone(),
        _ => panic!("weight {id} not f32"),
    }
}

fn load_into(w: &mut ModelWeights, net: &TrainableNet) {
    let g = layout::net_to_graph(net);
    set(w, "ftw", g.ftw);
    set(w, "ftb", g.ftb);
    set(w, "psqw", g.psqw);
    set(w, "psqb", vec![0.0; net.buckets]);
    set(w, "l1w", g.l1w);
    set(w, "l1b", g.l1b);
    set(w, "l2w", g.l2w);
    set(w, "l2b", g.l2b);
    set(w, "outw", g.outw);
    set(w, "outb", g.outb);
}

fn extract(w: &ModelWeights, base: &TrainableNet) -> TrainableNet {
    let g = GraphTensors {
        ftw: get(w, "ftw"),
        ftb: get(w, "ftb"),
        psqw: get(w, "psqw"),
        l1w: get(w, "l1w"),
        l1b: get(w, "l1b"),
        l2w: get(w, "l2w"),
        l2b: get(w, "l2b"),
        outw: get(w, "outw"),
        outb: get(w, "outb"),
    };
    let mut n = base.clone();
    layout::graph_to_net(&g, &mut n);
    n
}

fn lr_at(step: usize, total: usize, lr: f32, lr_min: f32, warmup: usize) -> f32 {
    if step < warmup {
        return lr * (step + 1) as f32 / warmup as f32;
    }
    let p = (step - warmup) as f32 / total.saturating_sub(warmup).max(1) as f32;
    lr_min + 0.5 * (lr - lr_min) * (1.0 + (std::f32::consts::PI * p.min(1.0)).cos())
}

fn load_init(spec: &str) -> Result<TrainableNet, String> {
    let bytes = if spec == "bundled" { BUNDLED_NNUE.to_vec() } else { std::fs::read(spec).map_err(|e| format!("{spec}: {e}"))? };
    TrainableNet::from_tnn_bytes(&bytes)
}

pub fn run(c: GpuCfg) -> Result<(), String> {
    let device: Arc<Device<_>> = DefaultDevice::new(0).map_err(|e| format!("{e:?}"))?;
    run_on(device, c)
}

fn run_on<G: Gpu>(device: Arc<Device<G>>, c: GpuCfg) -> Result<(), String> {
    let base = load_init(&c.init)?;
    if let Some(k) = c.holdout {
        if !c.d4 || k > 7 {
            return Err("--holdout-mode needs D4 on and a mode 0..7".into());
        }
        println!("[holdout] DIAGNOSTIC RUN: D4 orientation {k} is never trained, it is validated separately");
    }
    let (mapper, defn) = build(&base, c.d4, c.holdout, c.wsum_clip, c.proof_blend, c.result_weight, c.loss_weight);
    let mut weights = ModelWeights::new(&defn, c.seed);
    load_into(&mut weights, &base);
    let params = RangerParams { decay: 0.0, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98, alpha: 0.5, k: 6 };
    if let Some((muon_lr, mom)) = c.muon {
        // D1181/D1262/D1629: mandatory default: Muon (NS5) on dense hidden matrices l1w/l2w + Schedule-Free AdamW on all
        // other tensors, with Schedule-Free iterate averaging x across every tensor (validated and exported on x).
        let (nb, l1, l2) = (base.buckets, base.l1, base.l2);
        let din = base.l1_w.len() / (nb * l1).max(1);
        let shapes = vec![
            crate::gpumuon::MuonShape { name: "l1w".into(), size: nb * l1 * din, rows: l1, cols: din, buckets: nb },
            crate::gpumuon::MuonShape { name: "l2w".into(), size: nb * l2 * l1, rows: l2, cols: l1, buckets: nb },
        ];
        for s in &shapes {
            let same: Vec<String> = weights.iter().filter(|(_, v)| v.values.size() == s.size).map(|(id, _)| id.clone()).collect();
            if same != vec![s.name.clone()] {
                return Err(format!("--opt muon: tensor size {} of {} is not unique in the graph ({same:?})", s.size, s.name));
            }
        }
        let reg = crate::gpusf::SfReg::<G>::default();
        let sf = crate::gpusf::SfParams { reg: reg.clone(), ..Default::default() };
        println!("[opt] MUON + SCHEDULE-FREE ADAMW (D1181/D1554/D1629): Muon lr {muon_lr} (momentum {mom}, Nesterov, NS5) on {nb} blocks of l1w {l1}x{din} & l2w {l2}x{l1}; Schedule-Free AdamW (beta1 {} beta2 {} eps {} decay {} clip +-{} lr {} constant after warmup {}) on every other tensor; iterate average x evaluated & exported across ALL tensors", sf.beta1, sf.beta2, sf.eps, sf.decay, sf.max_weight, c.lr, c.warmup);
        let p = crate::gpumuon::MuonSfParams { sf, shapes, muon_lr, lr0: c.lr, mom };
        let opt = Optimiser::<G, crate::gpumuon::MuonSfAdamW<G>>::new(defn, weights, device, p).map_err(|e| format!("{e:?}"))?;
        return train_with(opt, c, base, mapper, Some(reg));
    }
    if c.sf {
        // EXPERIMENTAL ONLY (--opt sfadamw --experiment): Schedule-Free AdamW without Muon
        let reg = crate::gpusf::SfReg::<G>::default();
        let p = crate::gpusf::SfParams { reg: reg.clone(), ..Default::default() };
        println!("[opt] EXPERIMENT SCHEDULE-FREE ADAMW ONLY (NO MUON): beta1 {} beta2 {} eps {} decay {} clip +-{}; lr {} constant after warmup {} (no schedule); eval/export on the averaged iterate x", p.beta1, p.beta2, p.eps, p.decay, p.max_weight, c.lr, c.warmup);
        let opt = Optimiser::<G, crate::gpusf::SfAdamW<G>>::new(defn, weights, device, p).map_err(|e| format!("{e:?}"))?;
        return train_with(opt, c, base, mapper, Some(reg));
    }
    let opt = Optimiser::<G, Ranger<G>>::new(defn, weights, device, params).map_err(|e| format!("{e:?}"))?;
    train_with(opt, c, base, mapper, None)
}

/// the training loop of `run_on`, generic over the optimiser state (Ranger by default, Muon + Ranger with --opt muon)
fn train_with<G: Gpu, S: bullet_trainer::optimiser::OptimiserState<G>>(mut opt: Optimiser<G, S>, c: GpuCfg, base: TrainableNet, mapper: ModelInputsMapper<Rec>, sf_reg: Option<crate::gpusf::SfReg<G>>) -> Result<(), String> {
    std::fs::create_dir_all(&c.out).map_err(|e| e.to_string())?;
    let ckpt = format!("{}/bullet_ckpt", c.out);
    let mut start_sb = 1;
    if c.resume {
        if let Ok(t) = std::fs::read_to_string(format!("{}/gpu_state.txt", c.out)) {
            start_sb = t.trim().parse::<usize>().map_err(|e| e.to_string())? + 1;
            opt.load_from_checkpoint(&ckpt).map_err(|e| format!("{e:?}"))?;
            println!("[resume] superbatch {start_sb} (weights + optimiser state)");
        }
    }
    let mut valf = Tdat::open(&c.val)?;
    let mut val = trainer::load_val(&mut valf, c.val_rows)?;
    if let Some(rw) = c.result_weight {
        val = val.into_iter().map(|r| rw_val_row(r, rw)).collect();
        println!("[result-weight] {rw:?} by NNUE bucket (16; empties*16/50): val MSE is measured on this run's own target");
    }
    if let Some(lw) = c.loss_weight {
        println!("[loss-weight] {lw:?} by NNUE bucket (16)");
    }
    let sb = c.eval_every.max(1);
    let total_sb = (c.steps + sb - 1) / sb;
    let steps = TrainingSteps { batch_size: c.bs, batches_per_superbatch: sb, start_superbatch: start_sb, end_superbatch: total_sb };
    let (lr, lr_min, warmup, total) = (c.lr, c.lr_min, c.warmup, total_sb * sb);
    let lr_scale = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(1.0f32.to_bits()));
    let sc = lr_scale.clone();
    let schedule = TrainingSchedule {
        steps,
        log_rate: 100,
        // plateau scaling: the validation callback lowers this multiplier when val stalls (reduce-on-plateau)
        lr_schedule: Box::new(move |s| lr_at(s.total_batches(), total, lr, lr_min, warmup) * f32::from_bits(sc.load(std::sync::atomic::Ordering::Relaxed))),
        track_grad_norms: false,
    };
    let ram = if c.ram { Some(std::sync::Arc::new(Tdat::open_ram(&c.data)?)) } else { None };
    let loader = ReadMapLoader::new(TdatReader { path: c.data.clone(), seed: c.seed, ram }, mapper, c.threads.min(255) as u8);
    let (out, threads, ema_d, guard, holdout) = (c.out.clone(), c.threads, c.ema, c.guard, c.holdout);
    let (patience, min_delta) = (c.patience, c.min_delta);
    let (plateau, lr_factor, max_drops) = (c.plateau, c.lr_factor, c.max_drops);
    let keep_all = c.keep_all;
    let (val_stem, val_rows) = (c.val.clone(), c.val_rows);
    let mut since_best = 0usize;
    let mut drops = 0usize;
    // D1553 epoch rule: batches since the last val gain >= min_delta, and how many flat epochs already halved the LR
    let epoch_rule = c.epoch_rule;
    let epoch_batches = if epoch_rule { Tdat::open(&c.data)?.rows.div_ceil(c.bs).max(1) } else { 0 };
    if epoch_rule {
        println!("[epoch rule] LAWS C15/D5 (D1553): 1 epoch = {epoch_batches} batches of {} ({} rows); STOP after 4 epochs without a val gain >= {:e}", c.bs, epoch_batches * c.bs, c.min_delta);
    }
    let mut batches_flat = 0usize;
    let mut flat_epochs_done = 0usize;
    let mut best = f64::MAX;
    let mut last: Option<String> = None;
    let mut ema_net: Option<TrainableNet> = None;
    // fixed probe of TRAINING rows (same size as val): val - train gap = overfit signal
    let probe = {
        let mut t = trainer::Tdat::open(&c.data)?;
        trainer::load_val(&mut t, c.val_rows.min(16384))?
    };
    let init_mse = trainer::evaluate(&load_init(&c.init)?, &val, threads).mse;
    println!("[init] val mse {init_mse:.6} (guard {})", if guard > 0.0 { format!("{:.0}% of init", guard * 100.0) } else { "off".into() });
    train(&mut opt, schedule, loader, |_, _, _| {}, move |o, step| {
        let cpu = o.cpu_weights().expect("sync weights");
        // D1554: under Schedule-Free the averaged iterate x is the model (validated, checkpointed, exported); y is only the gradient point
        let net = match &sf_reg {
            Some(r) => extract(&crate::gpusf::averaged_weights(r, &cpu).expect("sfadamw averaged weights"), &base),
            None => extract(&cpu, &base),
        };
        drop(cpu);
        let sbn = step.total_batches() / sb;
        let v = trainer::evaluate(&net, &val, threads);
        let mut line = format!("[val superbatch {sbn}] live mse {:.6} proven {:.6} other {:.6}", v.mse, v.mse_proven, v.mse_other);
        let trs = trainer::evaluate(&net, &probe, threads);
        let tr = trs.mse;
        line += &format!(" | train-proven {:.6} (n={}; val has NO Zataxx proofs)", trs.mse_proven, trs.n_proven);
        // the GPU idles while this callback runs: keep the extra probes small and print their cost
        let t_probe = std::time::Instant::now();
        let modes = trainer::d4_modes(&net, &val, threads, 8192);
        let (dlo, dhi) = (modes.iter().cloned().fold(f64::MAX, f64::min), modes.iter().cloned().fold(0.0, f64::max));
        line += &format!(" | train-probe {:.6} gap {:+.6} | d4 val {:.6}..{:.6} spread {:.1}%", tr, v.mse - tr, dlo, dhi, 100.0 * (dhi - dlo) / dlo.max(1e-12));
        if let Some(k) = holdout {
            let rest: f64 = (0..8).filter(|&m| m != k).map(|m| modes[m]).sum::<f64>() / 7.0;
            line += &format!(" | HELD-OUT mode {k} {:.6} vs trained-modes {:.6} ({:+.1}%)", modes[k], rest, 100.0 * (modes[k] - rest) / rest.max(1e-12));
        }
        line += &format!(" | probes {:.1} s", t_probe.elapsed().as_secs_f64());
        if (dhi - dlo) / dlo.max(1e-12) > 0.05 {
            line += " | WARNING orientation spread >5%: net may be overfitting the stored orientation";
        }
        let (mut pick, mut pick_mse) = (net.clone(), v.mse);
        if ema_d > 0.0 {
            match ema_net.as_mut() {
                Some(e) => e.blend_from(&net, ema_d),
                None => ema_net = Some(net.clone()),
            }
            let ev = trainer::evaluate(ema_net.as_ref().unwrap(), &val, threads);
            line += &format!(" | ema mse {:.6}", ev.mse);
            if ev.mse < pick_mse {
                pick = ema_net.clone().unwrap();
                pick_mse = ev.mse;
            }
        }
        println!("{line}");
        // keep only the newest checkpoint + best.tnn (~28 MB each)
        let cur = format!("{out}/ckpt_{}.tnn", step.total_batches());
        let pick_bytes = pick.to_tnn_bytes();
        let _ = std::fs::write(&cur, &pick_bytes);
        crate::prov::record(&out, &format!("ckpt_{}.tnn", step.total_batches()), &pick_bytes, pick_mse, &val_stem, val_rows, sbn);
        if let Some(p) = last.replace(cur) {
            if !keep_all {
                let _ = std::fs::remove_file(p);
            }
        }
        let improved = pick_mse < best - min_delta;
        if pick_mse < best {
            // a gain smaller than min_delta is saved but does not reset the patience counter
            since_best = if pick_mse < best - min_delta { 0 } else { since_best + 1 };
            best = pick_mse;
            let _ = std::fs::write(format!("{out}/best.tnn"), &pick_bytes);
            crate::prov::record(&out, "best.tnn", &pick_bytes, pick_mse, &val_stem, val_rows, sbn);
        } else {
            since_best += 1;
        }
        let _ = std::fs::create_dir_all(&ckpt);
        if o.write_to_checkpoint(&ckpt).is_ok() {
            let _ = std::fs::write(format!("{out}/gpu_state.txt"), format!("{sbn}"));
        }
        if guard > 0.0 && v.mse > init_mse * guard as f64 {
            let msg = format!("GUARD TRIPPED at superbatch {sbn}: live val mse {:.6} > {:.2} x init {:.6}. Run aborted to save money (Mistake 39: lower the LR / check the data).", v.mse, guard, init_mse);
            eprintln!("{msg}");
            let _ = std::fs::write(format!("{out}/GUARD_TRIPPED.txt"), &msg);
            std::process::exit(3);
        }
        // D1553 epoch rule (default): per-epoch adaptive LR + stop after 4 epochs of data without a val gain >= min_delta
        if epoch_rule {
            if improved {
                batches_flat = 0;
                flat_epochs_done = 0;
            } else {
                batches_flat += sb;
            }
            let flat_epochs = batches_flat / epoch_batches;
            if flat_epochs >= 4 {
                let msg = format!("SATURATED (LAWS C15, D1553) at superbatch {sbn}: 4 epochs of data ({batches_flat} batches) without a val gain >= {min_delta:e}; best val mse {best:.6} in best.tnn.");
                println!("{msg}");
                let _ = std::fs::write(format!("{out}/EARLY_STOP.txt"), &msg);
                std::process::exit(0);
            }
            if flat_epochs > flat_epochs_done {
                flat_epochs_done = flat_epochs;
                println!("[epoch rule] {flat_epochs} epoch(s) of data without a val gain >= {min_delta:e} (val {pick_mse:.6}, best {best:.6}); stop at 4");
            }
        }
        // reduce-on-plateau: `plateau` checkpoints without a val gain -> lr x factor; after `max_drops` drops that still do not
        // help, stop (best.tnn keeps the best weights)
        if plateau > 0 && since_best >= plateau {
            if drops >= max_drops {
                let msg = format!("EARLY STOP at superbatch {sbn}: best val mse {best:.6}; {drops} lr drops (x{lr_factor} each, lr scale now {:.4}) did not help.", f32::from_bits(lr_scale.load(std::sync::atomic::Ordering::Relaxed)));
                println!("{msg}");
                let _ = std::fs::write(format!("{out}/EARLY_STOP.txt"), &msg);
                std::process::exit(0);
            }
            drops += 1;
            since_best = 0;
            let s = f32::from_bits(lr_scale.load(std::sync::atomic::Ordering::Relaxed)) * lr_factor;
            lr_scale.store(s.to_bits(), std::sync::atomic::Ordering::Relaxed);
            println!("[lr] PLATEAU at superbatch {sbn}: val mse {:.6} best {best:.6}. LR DROP {drops}/{max_drops}: lr scale -> {s:.4}", pick_mse);
        }
        // early stopping: val loss (the generalisation signal) has not improved for `patience` checkpoints -> stop, keep best.tnn
        if plateau == 0 && patience > 0 && since_best >= patience {
            let msg = format!("EARLY STOP at superbatch {sbn}: best val mse {best:.6} not improved by {min_delta:e} for {since_best} checkpoints ({} steps). best.tnn holds the best weights.", since_best * sb);
            println!("{msg}");
            let _ = std::fs::write(format!("{out}/EARLY_STOP.txt"), &msg);
            std::process::exit(0);
        }
    })
    .map_err(|_| "bullet training failed (TrainingError<G> has no Debug bound here; see the console log above)".to_string())?;
    Ok(())
}

/// `ti train gpu-parity --init net --tdat stem [--n 512]`: graph forward (bullet, on the GPU) vs hub CPU forward on the
/// same positions and the same weights. Proves the layout + graph before any training. Exit 1 if max diff > 1e-3.
pub fn parity(init: &str, stem: &str, n: usize) -> Result<(), String> {
    let device: Arc<Device<_>> = DefaultDevice::new(0).map_err(|e| format!("{e:?}"))?;
    parity_on(device, init, stem, n)
}

fn parity_on<G: Gpu>(device: Arc<Device<G>>, init: &str, stem: &str, n: usize) -> Result<(), String> {
    use bullet_trainer::model::ModelEvaluator;
    use bullet_trainer::run::{HostPool, Step};
    let net = load_init(init)?;
    let (mapper, defn) = build(&net, false, None, 0.0, 1.0, None, None);
    let mut w = ModelWeights::new(&defn, 1);
    load_into(&mut w, &net);
    let mut ev = ModelEvaluator::new(&defn, device.clone()).map_err(|e| format!("{e:?}"))?;
    ev.load_weights(&w).map_err(|e| format!("{e:?}"))?;
    let pool = HostPool::new(device.clone());
    let mut t = Tdat::open(stem)?;
    let rows = trainer::load_val(&mut t, n)?;
    let (mut worst, mut sum) = (0f32, 0f64);
    for r in rows.iter().take(n) {
        let host = mapper.map(&pool, std::slice::from_ref(r), Step::default(), 1).map_err(|e| format!("{e:?}"))?;
        let inputs = host.to_device(&device).map_err(|e| format!("{e:?}"))?;
        let out = ev.evaluate(&inputs).map_err(|e| format!("{e:?}"))?;
        let y = match out.get("output").ok_or("no output")?.to_host().map_err(|e| format!("{e:?}"))? {
            TValue::F32(v) => v[0],
            _ => return Err("output not f32".into()),
        };
        let d = (y - net.forward_y(&mapper::board_of(r, 0))).abs();
        worst = worst.max(d);
        sum += d as f64;
    }
    println!("gpu-parity rows {} max|graph - hub| {:.3e} mean {:.3e}", n.min(rows.len()), worst, sum / n.min(rows.len()).max(1) as f64);
    if worst < 1e-3 { Ok(()) } else { Err("GPU PARITY FAILED: graph != hub forward (check layout.rs / graph)".into()) }
}
