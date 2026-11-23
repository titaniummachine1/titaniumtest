//! D1181/D1262/D1629: `--opt muon` on `--backend cuda` = **Muon + Schedule-Free AdamW (`MuonSfAdamW`)**, the mandatory
//! default optimiser (LAWS D5/D1260).
//! - Muon (momentum, Nesterov, quintic Newton-Schulz `newton_schulz5`) updates the primary iterate `z` of every phase
//!   bucket block of the 2-D dense hidden matrices `l1w` (`l1 x din`) and `l2w` (`l2 x l1`);
//! - Schedule-Free AdamW (`SfAdamW`) updates the primary iterate `z` of every other tensor (`ftw`, `ftb`, `psqw`, `psqb`,
//!   `l1b`, `l2b`, `outw`, `outb`);
//! - Schedule-Free polynomial iterate averaging (`c = lr_t^2 / sum_{i<=t} lr_i^2`, `x += c * (z - x)`, gradient point
//!   `y = (1 - beta1) * z + beta1 * x`) runs across EVERY tensor (`MuonBlock` and `SfAdamW`), with every averaged iterate
//!   `x` registered in `SfReg` in sorted tensor-name order so validation, checkpoints and exports evaluate strictly on `x`.
use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::tensor::{DType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    runtime::{Device, Gpu, Stream},
};
use bullet_trainer::optimiser::OptimiserState;

use crate::gpusf::{SfAdamW, SfParams};

/// A tensor updated by Muon: its flat size (used to recognise it: sizes are checked unique at setup), the graph layout
/// (column-major, `rows_total` = buckets x rows) and one block = `rows` x `cols`.
#[derive(Clone, Debug)]
pub struct MuonShape {
    pub name: String,
    pub size: usize,
    pub rows: usize,
    pub cols: usize,
    pub buckets: usize,
}

pub struct MuonSfParams<G: Gpu> {
    pub sf: SfParams<G>,
    pub shapes: Vec<MuonShape>,
    /// Muon lr at schedule factor 1; the live factor = current lr / `lr0`
    pub muon_lr: f32,
    pub lr0: f32,
    pub mom: f32,
}

impl<G: Gpu> Clone for MuonSfParams<G> {
    fn clone(&self) -> Self {
        MuonSfParams { sf: self.sf.clone(), shapes: self.shapes.clone(), muon_lr: self.muon_lr, lr0: self.lr0, mom: self.mom }
    }
}

impl<G: Gpu> std::fmt::Debug for MuonSfParams<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MuonSfParams(muon_lr {} lr0 {} mom {} shapes {} {:?})", self.muon_lr, self.lr0, self.mom, self.shapes.len(), self.sf)
    }
}

impl<G: Gpu> Default for MuonSfParams<G> {
    fn default() -> Self {
        MuonSfParams { sf: SfParams::default(), shapes: Vec::new(), muon_lr: 1e-3, lr0: 1e-3, mom: 0.95 }
    }
}

pub struct MuonBlock<G: Gpu> {
    shape: MuonShape,
    m: Vec<f32>,
    z: Vec<f32>,
    x_host: Vec<f32>,
    x_dev: Arc<Buffer<G>>,
    t: u64,
    sum_lr2: f64,
    host_w: TValue,
    host_x: TValue,
    p: MuonSfParams<G>,
}

pub enum MuonSfAdamW<G: Gpu> {
    R(SfAdamW<G>),
    M(MuonBlock<G>),
}

fn f32s(v: TValue) -> Vec<f32> {
    match v {
        TValue::F32(x) => x,
        _ => panic!("muon: expected an f32 buffer"),
    }
}

/// Pure host Schedule-Free + Muon step on a column-major dense block: updates `z` via Muon (`step_colmajor`),
/// then updates averaged iterate `x += c * (z - x)` (`c = lr_avg^2 / sum_lr2`) and writes `y_out = (1 - beta1) * z + beta1 * x`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_sf_colmajor(
    shape: &MuonShape,
    m: &mut [f32],
    z: &mut [f32],
    x: &mut [f32],
    y_out: &mut [f32],
    g: &[f32],
    gf: f32,
    lr_muon: f32,
    lr_avg: f64,
    sum_lr2: &mut f64,
    beta1: f32,
    mu: f32,
    lo: f32,
    hi: f32,
) {
    step_colmajor(shape, m, g, z, gf, lr_muon, mu, lo, hi);
    *sum_lr2 += lr_avg * lr_avg;
    let c = if *sum_lr2 > 0.0 { (lr_avg * lr_avg / *sum_lr2) as f32 } else { 1.0 };
    for i in 0..z.len() {
        x[i] += c * (z[i] - x[i]);
        y_out[i] = (1.0 - beta1) * z[i] + beta1 * x[i];
    }
}

impl<G: Gpu> MuonBlock<G> {
    fn step(&mut self, g: &[f32], y_cur: &[f32], gf: f32, lr_now: f32) -> Vec<f32> {
        self.t += 1;
        if self.t == 1 {
            self.z.copy_from_slice(y_cur);
            self.x_host.copy_from_slice(y_cur);
        }
        let lr_muon = self.p.muon_lr * lr_now / self.p.lr0.max(1e-12);
        let mut y_next = vec![0f32; self.z.len()];
        step_sf_colmajor(
            &self.shape,
            &mut self.m,
            &mut self.z,
            &mut self.x_host,
            &mut y_next,
            g,
            gf,
            lr_muon,
            lr_now as f64,
            &mut self.sum_lr2,
            self.p.sf.beta1,
            self.p.mom,
            self.p.sf.min_weight,
            self.p.sf.max_weight,
        );
        y_next
    }
}

/// one Muon step on a column-major graph tensor (block b = graph rows [b*rows, (b+1)*rows) x all cols, index (r, c) =
/// c * buckets*rows + r); `m` = momentum in the same layout; the same math as train.rs Muon::step_blocks (row-major)
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_colmajor(s: &MuonShape, m_all: &mut [f32], g: &[f32], w: &mut [f32], gf: f32, lr: f32, mu: f32, lo: f32, hi: f32) {
    {
        let (rows, cols, rt) = (s.rows, s.cols, s.buckets * s.rows);
        let scale = ((rows as f32) / (cols as f32)).max(1.0).sqrt();
        let blocks: Vec<(usize, Vec<f32>)> = std::thread::scope(|sc| {
            let m = &*m_all;
            let hs: Vec<_> = (0..s.buckets)
                .map(|b| {
                    sc.spawn(move || {
                        let idx = |i: usize, j: usize| j * rt + b * rows + i;
                        let mut gi = vec![0f32; rows * cols];
                        let mut any = false;
                        for i in 0..rows {
                            for j in 0..cols {
                                let k = idx(i, j);
                                gi[i * cols + j] = g[k] * gf;
                                any |= g[k] != 0.0 || m[k] != 0.0;
                            }
                        }
                        if !any {
                            return (b, Vec::new()); // a bucket no sample reached and no momentum left
                        }
                        let mut mb = vec![0f32; rows * cols];
                        let mut u = vec![0f32; rows * cols];
                        for i in 0..rows {
                            for j in 0..cols {
                                let k = idx(i, j);
                                let mm = mu * m[k] + gi[i * cols + j];
                                mb[i * cols + j] = mm;
                                u[i * cols + j] = gi[i * cols + j] + mu * mm; // Nesterov
                            }
                        }
                        let o = crate::train::newton_schulz5(&u, rows, cols);
                        let mut out = mb;
                        out.extend(o);
                        (b, out)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("muon block thread")).collect()
        });
        for (b, v) in blocks {
            if v.is_empty() {
                continue;
            }
            let (mb, o) = v.split_at(rows * cols);
            for i in 0..rows {
                for j in 0..cols {
                    let k = j * rt + b * rows + i;
                    m_all[k] = mb[i * cols + j];
                    w[k] = (w[k] - lr * scale * o[i * cols + j]).clamp(lo, hi);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{step_colmajor, step_sf_colmajor, MuonShape};
    /// the column-major GPU step must equal a row-major reference (CPU MuonAdamW math: m = mu m + g; u = g + mu m;
    /// w -= lr * max(1, r/c)^0.5 * NS5(u)) block by block, over two steps (momentum carried)
    #[test]
    fn colmajor_matches_rowmajor() {
        let (nb, rows, cols) = (3usize, 4usize, 6usize);
        let rt = nb * rows;
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let n = nb * rows * cols;
        let w_row: Vec<f32> = (0..n).map(|_| rnd() * 0.1).collect(); // row-major (nb*rows) x cols
        let to_col = |v: &[f32]| -> Vec<f32> {
            let mut o = vec![0f32; n];
            for r in 0..rt {
                for c in 0..cols {
                    o[c * rt + r] = v[r * cols + c];
                }
            }
            o
        };
        let sh = MuonShape { name: "t".into(), size: n, rows, cols, buckets: nb };
        let (gf, lr, mu) = (0.5f32, 0.01f32, 0.95f32);
        let (mut wr, mut mr) = (w_row.clone(), vec![0f32; n]);
        let (mut wc, mut mc) = (to_col(&w_row), vec![0f32; n]);
        for _ in 0..2 {
            let g_row: Vec<f32> = (0..n).map(|_| rnd()).collect();
            // row-major reference
            let scale = ((rows as f32) / (cols as f32)).max(1.0).sqrt();
            for b in 0..nb {
                let r0 = b * rows * cols;
                let mut u = vec![0f32; rows * cols];
                for i in 0..rows * cols {
                    let gi = g_row[r0 + i] * gf;
                    mr[r0 + i] = mu * mr[r0 + i] + gi;
                    u[i] = gi + mu * mr[r0 + i];
                }
                let o = crate::train::newton_schulz5(&u, rows, cols);
                for i in 0..rows * cols {
                    wr[r0 + i] = (wr[r0 + i] - lr * scale * o[i]).clamp(-1.98, 1.98);
                }
            }
            step_colmajor(&sh, &mut mc, &to_col(&g_row), &mut wc, gf, lr, mu, -1.98, 1.98);
        }
        let want = to_col(&wr);
        for k in 0..n {
            assert!((want[k] - wc[k]).abs() < 1e-6, "index {k}: {} vs {}", want[k], wc[k]);
        }
        assert!(want.iter().zip(&to_col(&w_row)).any(|(a, b)| (a - b).abs() > 1e-4), "weights must move");
    }

    /// Schedule-Free + Muon step (`step_sf_colmajor`) must equal row-major Muon on `z` followed by `x` averaging and `y` interpolation.
    #[test]
    fn sf_muon_step_matches_cpu_reference() {
        let (nb, rows, cols) = (2usize, 4usize, 6usize);
        let rt = nb * rows;
        let n = nb * rows * cols;
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let to_col = |v: &[f32]| -> Vec<f32> {
            let mut o = vec![0f32; n];
            for r in 0..rt {
                for c in 0..cols {
                    o[c * rt + r] = v[r * cols + c];
                }
            }
            o
        };
        let w_row: Vec<f32> = (0..n).map(|_| rnd() * 0.1).collect();
        let sh = MuonShape { name: "t".into(), size: n, rows, cols, buckets: nb };
        let (gf, lr_muon, mu, b1) = (1.0f32, 0.01f32, 0.95f32, 0.9f32);
        let mut zr = w_row.clone();
        let mut xr = w_row.clone();
        let mut yr = w_row.clone();
        let mut mr = vec![0f32; n];
        let mut zc = to_col(&w_row);
        let mut xc = to_col(&w_row);
        let mut yc = to_col(&w_row);
        let mut mc = vec![0f32; n];
        let mut sum_lr2_r = 0.0f64;
        let mut sum_lr2_c = 0.0f64;
        for step in 1..=2 {
            let lr_avg = if step == 1 { 0.0005 } else { 0.001 };
            let g_row: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let scale = ((rows as f32) / (cols as f32)).max(1.0).sqrt();
            for b in 0..nb {
                let r0 = b * rows * cols;
                let mut u = vec![0f32; rows * cols];
                for i in 0..rows * cols {
                    let gi = g_row[r0 + i] * gf;
                    mr[r0 + i] = mu * mr[r0 + i] + gi;
                    u[i] = gi + mu * mr[r0 + i];
                }
                let o = crate::train::newton_schulz5(&u, rows, cols);
                for i in 0..rows * cols {
                    zr[r0 + i] = (zr[r0 + i] - lr_muon * scale * o[i]).clamp(-1.98, 1.98);
                }
            }
            sum_lr2_r += lr_avg * lr_avg;
            let c = (lr_avg * lr_avg / sum_lr2_r) as f32;
            for i in 0..n {
                xr[i] += c * (zr[i] - xr[i]);
                yr[i] = (1.0 - b1) * zr[i] + b1 * xr[i];
            }
            step_sf_colmajor(&sh, &mut mc, &mut zc, &mut xc, &mut yc, &to_col(&g_row), gf, lr_muon, lr_avg, &mut sum_lr2_c, b1, mu, -1.98, 1.98);
        }
        let want_x = to_col(&xr);
        let want_y = to_col(&yr);
        for k in 0..n {
            assert!((want_x[k] - xc[k]).abs() < 1e-6, "x[{k}]: {} vs {}", want_x[k], xc[k]);
            assert!((want_y[k] - yc[k]).abs() < 1e-6, "y[{k}]: {} vs {}", want_y[k], yc[k]);
        }
        assert!(xc.iter().zip(&to_col(&w_row)).any(|(a, b)| (a - b).abs() > 1e-4));
    }
}

impl<G: Gpu> OptimiserState<G> for MuonSfAdamW<G> {
    type Params = MuonSfParams<G>;

    fn new(device: &Arc<Device<G>>, size: usize, params: Self::Params) -> Result<Self, G::Error> {
        match params.shapes.iter().find(|s| s.size == size).cloned() {
            Some(shape) => {
                let zeros = TValue::zeros(DType::F32, size);
                let x_dev = Buffer::from_host(device, &zeros)?;
                params.sf.reg.0.lock().unwrap().push(x_dev.clone());
                Ok(MuonSfAdamW::M(MuonBlock {
                    shape,
                    m: vec![0.0; size],
                    z: vec![0.0; size],
                    x_host: vec![0.0; size],
                    x_dev,
                    t: 0,
                    sum_lr2: 0.0,
                    host_w: zeros.clone(),
                    host_x: zeros,
                    p: params,
                }))
            }
            None => Ok(MuonSfAdamW::R(<SfAdamW<G> as OptimiserState<G>>::new(device, size, params.sf)?)),
        }
    }

    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> Result<bullet_trainer::optimiser::OptimiserUpdateSync<'a, G>, G::Error> {
        match self {
            MuonSfAdamW::R(r) => r.update(stream, weights, grads, gradient_factor, learning_rate),
            MuonSfAdamW::M(mb) => {
                // copies on the compute stream run after the backward pass that wrote the gradient
                let g = f32s(grads.to_host_async(stream)?.value()?);
                let w = f32s(weights.to_host_async(stream)?.value()?);
                let gf = f32s(gradient_factor.to_host_async(stream)?.value()?)[0];
                let lr_now = f32s(learning_rate.to_host_async(stream)?.value()?)[0];
                let y_next = mb.step(&g, &w, gf, lr_now);
                mb.host_w = TValue::F32(y_next);
                mb.host_x = TValue::F32(mb.x_host.clone());
                let mbr: &'a MuonBlock<G> = mb;
                let mut sync = bullet_trainer::optimiser::OptimiserUpdateSync::default();
                sync.push_copy(weights.copy_from_host_async(stream, &mbr.host_w)?);
                sync.push_copy(mbr.x_dev.copy_from_host_async(stream, &mbr.host_x)?);
                Ok(sync)
            }
        }
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        match self {
            MuonSfAdamW::R(r) => r.reset(),
            MuonSfAdamW::M(mb) => {
                mb.m.fill(0.0);
                mb.z.fill(0.0);
                mb.x_host.fill(0.0);
                mb.t = 0;
                mb.sum_lr2 = 0.0;
                mb.x_dev.copy_from_host(&TValue::zeros(DType::F32, mb.x_host.len()))
            }
        }
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        match self {
            MuonSfAdamW::R(r) => r.set_params(params.sf),
            MuonSfAdamW::M(mb) => {
                mb.p = params;
                Ok(())
            }
        }
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let mut rmap: BTreeMap<String, &mut SfAdamW<G>> = BTreeMap::new();
        for (id, s) in map.iter_mut() {
            match &mut **s {
                MuonSfAdamW::R(r) => {
                    rmap.insert(id.clone(), r);
                }
                MuonSfAdamW::M(mb) => {
                    for (tag, dst) in [("muon", &mut mb.m), ("sf_z", &mut mb.z), ("sf_x", &mut mb.x_host)] {
                        if let Ok(b) = std::fs::read(format!("{path}/{tag}_{id}.bin")) {
                            if b.len() == dst.len() * 4 {
                                *dst = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            }
                        }
                    }
                    mb.x_dev.copy_from_host(&TValue::F32(mb.x_host.clone()))?;
                    if let Ok(t) = std::fs::read_to_string(format!("{path}/sf_t_{id}.txt")) {
                        let mut it = t.split_whitespace();
                        mb.t = it.next().and_then(|a| a.parse().ok()).unwrap_or(0);
                        mb.sum_lr2 = it.next().and_then(|a| a.parse().ok()).unwrap_or(0.0);
                    }
                }
            }
        }
        <SfAdamW<G> as OptimiserState<G>>::load_from_checkpoint(&mut rmap, path)
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let mut rmap: BTreeMap<String, &SfAdamW<G>> = BTreeMap::new();
        for (id, s) in map.iter() {
            match s {
                MuonSfAdamW::R(r) => {
                    rmap.insert(id.clone(), r);
                }
                MuonSfAdamW::M(mb) => {
                    for (tag, src) in [("muon", &mb.m), ("sf_z", &mb.z), ("sf_x", &mb.x_host)] {
                        let b: Vec<u8> = src.iter().flat_map(|x| x.to_le_bytes()).collect();
                        let _ = std::fs::write(format!("{path}/{tag}_{id}.bin"), b);
                    }
                    let _ = std::fs::write(format!("{path}/sf_t_{id}.txt"), format!("{} {}", mb.t, mb.sum_lr2));
                }
            }
        }
        <SfAdamW<G> as OptimiserState<G>>::write_to_checkpoint(&rmap, path)
    }
}
