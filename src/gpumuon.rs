//! D1181 (user: port Muon to the GPU trainer): `--opt muon` on `--backend cuda`. Muon (momentum, Nesterov, quintic
//! Newton-Schulz, the CPU MuonAdamW math of train.rs D1159, same `newton_schulz5`) on every phase bucket block of the
//! two DENSE hidden matrices l1w (l1 x 2w per bucket) and l2w (l2 x l1 per bucket); bullet's Ranger (the standard GPU
//! optimiser, same parameters as the default run) keeps every other tensor. Without `--opt muon` the GPU trainer uses
//! `Ranger<G>` directly and never touches this file (default path unchanged).
//!
//! The Muon blocks are small (16 x 512 and 32 x 16 per bucket), so the update runs on the host: per step the gradient and
//! the weights of l1w / l2w are copied to the host (after the backward pass on the same stream), the Muon step is done
//! there (blocks in parallel), and the new weights are copied back on the same stream. Weight clip = Ranger's +-1.98.
use std::{collections::BTreeMap, sync::Arc};

use bullet_compiler::tensor::TValue;
use bullet_gpu::{
    buffer::Buffer,
    runtime::{Device, Gpu, Stream},
};
use bullet_trainer::optimiser::{
    ranger::{Ranger, RangerParams},
    OptimiserState,
};

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

#[derive(Clone, Debug, Default)]
pub struct MuonRangerParams {
    pub ranger: RangerParams,
    pub shapes: Vec<MuonShape>,
    /// Muon lr at schedule factor 1; the live factor = current lr / `lr0` (the same schedule as Ranger, D1164)
    pub muon_lr: f32,
    pub lr0: f32,
    pub mom: f32,
}

pub struct MuonBlock<G: Gpu> {
    shape: MuonShape,
    m: Vec<f32>,
    host_w: TValue,
    p: MuonRangerParams,
    _g: std::marker::PhantomData<G>,
}

pub enum MuonRanger<G: Gpu> {
    R(Ranger<G>),
    M(MuonBlock<G>),
}

fn f32s(v: TValue) -> Vec<f32> {
    match v {
        TValue::F32(x) => x,
        _ => panic!("muon: expected an f32 buffer"),
    }
}

impl<G: Gpu> MuonBlock<G> {
    fn step(&mut self, g: &[f32], w: &mut [f32], gf: f32, lr: f32) {
        step_colmajor(&self.shape, &mut self.m, g, w, gf, lr, self.p.mom, self.p.ranger.min_weight, self.p.ranger.max_weight);
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
    use super::{step_colmajor, MuonShape};
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
}

impl<G: Gpu> OptimiserState<G> for MuonRanger<G> {
    type Params = MuonRangerParams;

    fn new(device: &Arc<Device<G>>, size: usize, params: Self::Params) -> Result<Self, G::Error> {
        match params.shapes.iter().find(|s| s.size == size).cloned() {
            Some(shape) => Ok(MuonRanger::M(MuonBlock {
                m: vec![0.0; size],
                host_w: TValue::F32(vec![0.0; size]),
                shape,
                p: params,
                _g: std::marker::PhantomData,
            })),
            None => Ok(MuonRanger::R(<Ranger<G> as OptimiserState<G>>::new(device, size, params.ranger)?)),
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
            MuonRanger::R(r) => r.update(stream, weights, grads, gradient_factor, learning_rate),
            MuonRanger::M(mb) => {
                // copies on the compute stream run after the backward pass that wrote the gradient
                let g = f32s(grads.to_host_async(stream)?.value()?);
                let mut w = f32s(weights.to_host_async(stream)?.value()?);
                let gf = f32s(gradient_factor.to_host_async(stream)?.value()?)[0];
                let lr_now = f32s(learning_rate.to_host_async(stream)?.value()?)[0];
                let lr = mb.p.muon_lr * lr_now / mb.p.lr0.max(1e-12);
                mb.step(&g, &mut w, gf, lr);
                mb.host_w = TValue::F32(w);
                let mbr: &'a MuonBlock<G> = mb;
                let mut sync = bullet_trainer::optimiser::OptimiserUpdateSync::default();
                sync.push_copy(weights.copy_from_host_async(stream, &mbr.host_w)?);
                Ok(sync)
            }
        }
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        match self {
            MuonRanger::R(r) => r.reset(),
            MuonRanger::M(mb) => {
                mb.m.iter_mut().for_each(|x| *x = 0.0);
                Ok(())
            }
        }
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        match self {
            MuonRanger::R(r) => r.set_params(params.ranger),
            MuonRanger::M(mb) => {
                mb.p = params;
                Ok(())
            }
        }
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let mut rmap: BTreeMap<String, &mut Ranger<G>> = BTreeMap::new();
        for (id, s) in map.iter_mut() {
            match &mut **s {
                MuonRanger::R(r) => {
                    rmap.insert(id.clone(), r);
                }
                MuonRanger::M(mb) => {
                    if let Ok(b) = std::fs::read(format!("{path}/muon_{id}.bin")) {
                        if b.len() == mb.m.len() * 4 {
                            mb.m = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        }
                    }
                }
            }
        }
        <Ranger<G> as OptimiserState<G>>::load_from_checkpoint(&mut rmap, path)
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let mut rmap: BTreeMap<String, &Ranger<G>> = BTreeMap::new();
        for (id, s) in map.iter() {
            match s {
                MuonRanger::R(r) => {
                    rmap.insert(id.clone(), r);
                }
                MuonRanger::M(mb) => {
                    let b: Vec<u8> = mb.m.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let _ = std::fs::write(format!("{path}/muon_{id}.bin"), b);
                }
            }
        }
        <Ranger<G> as OptimiserState<G>>::write_to_checkpoint(&rmap, path)
    }
}
