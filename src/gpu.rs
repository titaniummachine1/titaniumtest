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
}

impl DataReader<Rec> for TdatReader {
    fn read_chunks<F: FnMut(&[Rec]) -> bool>(&self, skip_count: usize, mut f: F) {
        let mut t = Tdat::open(&self.path).expect("open tdat");
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
}

const NB: usize = 8;

/// Inputs, mapper and graph. Weight ids: ftw/ftb, psqw/psqb (psqb cancels in stm-ntm), l1w/l1b, l2w/l2b, outw/outb.
fn build(net: &TrainableNet, d4: bool) -> (ModelInputsMapper<Rec>, ModelDefinition) {
    let (w, l1, l2, oi) = (net.width, net.l1, net.l2, net.out_in());
    assert!(net.has_dis && !net.pairwise && net.v2 && net.buckets == NB && net.num_feats == NUM_FEATS, "graph covers TNN3, non-pairwise, 8 buckets");
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
        wt[0] = r.wsum.max(0.0);
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
        let o = ao.forward(h1.concat(h2)).select(bucket);
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
    let (mapper, defn) = build(&base, c.d4);
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
    let loader = ReadMapLoader::new(TdatReader { path: c.data.clone(), seed: c.seed }, mapper, c.threads.min(255) as u8);
    let (out, threads) = (c.out.clone(), c.threads);
    let mut best = f64::MAX;
    train(&mut opt, schedule, loader, |_, _, _| {}, move |o, step| {
        let cpu = o.cpu_weights().expect("sync weights");
        let net = extract(&cpu, &base);
        drop(cpu);
        let v = trainer::evaluate(&net, &val, threads);
        println!("[val superbatch {}] mse {:.6} proven {:.6} other {:.6}", step.total_batches() / sb, v.mse, v.mse_proven, v.mse_other);
        let _ = std::fs::write(format!("{out}/ckpt_{}.tnn", step.total_batches()), net.to_tnn_bytes());
        if v.mse < best {
            best = v.mse;
            let _ = std::fs::write(format!("{out}/best.tnn"), net.to_tnn_bytes());
        }
        let _ = std::fs::create_dir_all(&ckpt);
        if o.write_to_checkpoint(&ckpt).is_ok() {
            let _ = std::fs::write(format!("{out}/gpu_state.txt"), format!("{}", step.total_batches() / sb));
        }
    })
    .map_err(|_| "bullet training failed (TrainingError<G> has no Debug bound here; see the console log above)".to_string())?;
    Ok(())
}
