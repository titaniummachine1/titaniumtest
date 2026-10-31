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
use crate::targets::target_of;
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
    pub resume: bool,
    pub ram: bool,
    /// cap on the per-row loss weight (0 = off); Opus review: max wsum is 37.7, mean 1.2
    pub wsum_clip: f32,
    /// EMA of the weights over superbatches (0 = off): ema = d*ema + (1-d)*live, validated next to the live weights
    pub ema: f32,
    /// abort when val mse > guard * initial val mse (0 = off). Mistake 39 / D153: a bad fine-tune showed 0.0204 -> 0.0216 in epoch 1
    pub guard: f32,
}

const NB: usize = 8;

/// Inputs, mapper and graph. Weight ids: ftw/ftb, psqw/psqb (psqb cancels in stm-ntm), l1w/l1b, l2w/l2b, outw/outb.
fn build(net: &TrainableNet, d4: bool, wsum_clip: f32) -> (ModelInputsMapper<Rec>, ModelDefinition) {
    let (w, l1, l2, oi) = (net.width, net.l1, net.l2, net.out_in());
    assert!(!net.pairwise && net.s57w.is_empty() && net.buckets == NB && net.num_feats == NUM_FEATS,
        "graph covers non-pairwise nets without the S57 skip, 8 buckets, 26196 features (TNN1/TNN3)");
    let skip = net.v2; // TNN2: the output reads h1 || h2; TNN1/TNN3: h2 only (out_in = l2)
    let inputs = ModelInputs::default()
        .add_sparse("stm", (NUM_FEATS, 1), MAX_ACTIVE)
        .add_sparse("ntm", (NUM_FEATS, 1), MAX_ACTIVE)
        .add_sparse("bucket", (NB, 1), 1)
        .add_dense("target", (1, 1))
        .add_dense("wt", (1, 1));

    let mapper = ModelInputsMapper::build(&inputs, move |r: &Rec, step, ((((stm, ntm), bk), target), wt)| {
        let mode = if d4 {
            let h = (r.occ0 ^ r.occ1.rotate_left(23)).wrapping_add(step.total_batches() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            ((h >> 40) & 7) as usize
        } else {
            0
        };
        let s = mapper::sample(r, mode, NB);
        for (dst, src) in [(&mut *stm, &s.us), (&mut *ntm, &s.them)] {
            for (i, &f) in src.iter().enumerate() {
                dst[i] = f as i32;
            }
            if src.len() < MAX_ACTIVE {
                dst[src.len()] = -1;
            }
        }
        bk[0] = s.bucket as i32;
        target[0] = target_of(r).target;
        wt[0] = if wsum_clip > 0.0 { r.wsum.clamp(0.0, wsum_clip) } else { r.wsum.max(0.0) };
    });

    let defn = ModelDefinition::build(&inputs, move |b, ((((stm, ntm), bucket), target), wt)| {
        let ft = b.new_affine("ft", NUM_FEATS, w);
        let ps = b.new_affine("psq", NUM_FEATS, NB);
        let a1 = b.new_affine("l1", 2 * w, NB * l1);
        let a2 = b.new_affine("l2", l1, NB * l2);
        let ao = b.new_affine("out", oi, NB);

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
    set(w, "psqb", vec![0.0; NB]);
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
    let (mapper, defn) = build(&base, c.d4, c.wsum_clip);
    let mut weights = ModelWeights::new(&defn, c.seed);
    load_into(&mut weights, &base);
    let params = RangerParams { decay: 0.0, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98, alpha: 0.5, k: 6 };
    let mut opt = Optimiser::<G, Ranger<G>>::new(defn, weights, device, params).map_err(|e| format!("{e:?}"))?;
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
    let val = trainer::load_val(&mut valf, c.val_rows)?;
    let sb = c.eval_every.max(1);
    let total_sb = (c.steps + sb - 1) / sb;
    let steps = TrainingSteps { batch_size: c.bs, batches_per_superbatch: sb, start_superbatch: start_sb, end_superbatch: total_sb };
    let (lr, lr_min, warmup, total) = (c.lr, c.lr_min, c.warmup, total_sb * sb);
    let schedule = TrainingSchedule {
        steps,
        log_rate: 100,
        lr_schedule: Box::new(move |s| lr_at(s.total_batches(), total, lr, lr_min, warmup)),
        track_grad_norms: false,
    };
    let ram = if c.ram { Some(std::sync::Arc::new(Tdat::open_ram(&c.data)?)) } else { None };
    let loader = ReadMapLoader::new(TdatReader { path: c.data.clone(), seed: c.seed, ram }, mapper, c.threads.min(255) as u8);
    let (out, threads, ema_d, guard) = (c.out.clone(), c.threads, c.ema, c.guard);
    let mut best = f64::MAX;
    let mut last: Option<String> = None;
    let mut ema_net: Option<TrainableNet> = None;
    let init_mse = trainer::evaluate(&load_init(&c.init)?, &val, threads).mse;
    println!("[init] val mse {init_mse:.6} (guard {})", if guard > 0.0 { format!("{:.0}% of init", guard * 100.0) } else { "off".into() });
    train(&mut opt, schedule, loader, |_, _, _| {}, move |o, step| {
        let cpu = o.cpu_weights().expect("sync weights");
        let net = extract(&cpu, &base);
        drop(cpu);
        let sbn = step.total_batches() / sb;
        let v = trainer::evaluate(&net, &val, threads);
        let mut line = format!("[val superbatch {sbn}] live mse {:.6} proven {:.6} other {:.6}", v.mse, v.mse_proven, v.mse_other);
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
        let _ = std::fs::write(&cur, pick.to_tnn_bytes());
        if let Some(p) = last.replace(cur) {
            let _ = std::fs::remove_file(p);
        }
        if pick_mse < best {
            best = pick_mse;
            let _ = std::fs::write(format!("{out}/best.tnn"), pick.to_tnn_bytes());
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
    let (mapper, defn) = build(&net, false, 0.0);
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
