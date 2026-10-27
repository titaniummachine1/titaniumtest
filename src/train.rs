//! Bullet-style Native Rust Sparse-Feature NNUE Trainer for Titanium Ataxx (`TNN1` / `TNN2`).
//!
//! Eliminates Python/PyTorch and 13 GB `.idx` feature dump files by:
//! 1. Streaming 25-byte `Board` records (`dis_positions.bin`) + chunk labels (`c_<idx>.txt`) directly.
//! 2. Computing `nnfeat::both` on the fly across worker threads (`~150 ns/position`).
//! 3. Running data-parallel forward + backward passes for `|sigmoid(y) - target|^2.5` (Stockfish power loss).
//! 4. Performing Lazy Sparse AdamW updates on `ft` and `psqt` so only active feature rows touched
//!    in a mini-batch are updated, with `[-1.98, +1.98]` quantization-safe weight clamping.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::Instant;
use titanium::nnfeat::{self, NUM_FEATS};
use titanium::nnue::BUNDLED_NNUE;
use titanium::Board;

const QA_CLAMP: f32 = 1.98;

#[derive(Clone)]
pub struct TrainableNet {
    pub num_feats: usize,
    pub width: usize,
    pub l1: usize,
    pub l2: usize,
    pub buckets: usize,
    pub groups: u32,
    pub v2: bool,
    pub pairwise: bool,
    pub ft: Vec<f32>,
    pub ftb: Vec<f32>,
    pub psqt: Vec<f32>,
    pub l1_w: Vec<f32>,
    pub l1_b: Vec<f32>,
    pub l2_w: Vec<f32>,
    pub l2_b: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    pub s57w: Vec<f32>,
}

#[derive(Clone, Copy)]
pub struct LabeledSample {
    pub occ0: u64,
    pub occ1: u64,
    pub blk: u64,
    pub stm: u8,
    pub target_prob: f32,
}

impl TrainableNet {
    pub fn from_tnn_bytes(d: &[u8]) -> Result<Self, String> {
        let v2 = d.len() >= 4 && &d[..4] == b"TNN2";
        if d.len() < 28 || (&d[..4] != b"TNN1" && !v2) {
            return Err("not a TNN1/TNN2 file".to_string());
        }
        let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        let (num_feats, width, l1, l2, buckets, groups) =
            (u(0), u(1), u(2), u(3), u(4), u(5) as u32);
        let pairwise = v2 && u(6) != 0;
        if num_feats != NUM_FEATS {
            return Err(format!("num_feats {num_feats} != {NUM_FEATS}"));
        }
        let din = if pairwise { width } else { 2 * width };
        let out_in = if v2 { l1 + l2 } else { l2 };
        let mut p = if v2 { 32 } else { 28 };
        let mut take = |n: usize| -> Vec<f32> {
            let v = d[p..p + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            p += 4 * n;
            v
        };
        let net = TrainableNet {
            ft: take(num_feats * width),
            ftb: take(width),
            psqt: take(num_feats * buckets),
            l1_w: take(buckets * l1 * din),
            l1_b: take(buckets * l1),
            l2_w: take(buckets * l2 * l1),
            l2_b: take(buckets * l2),
            out_w: take(buckets * out_in),
            out_b: take(buckets),
            s57w: if v2 { take(buckets) } else { Vec::new() },
            num_feats,
            width,
            l1,
            l2,
            buckets,
            groups,
            v2,
            pairwise,
        };
        if p != d.len() {
            return Err("trailing or missing bytes in TNN file".to_string());
        }
        Ok(net)
    }

    pub fn to_tnn_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.ft.len() * 4 + 65536);
        out.extend_from_slice(if self.v2 { b"TNN2" } else { b"TNN1" });
        for &val in &[
            self.num_feats as u32,
            self.width as u32,
            self.l1 as u32,
            self.l2 as u32,
            self.buckets as u32,
            self.groups,
        ] {
            out.extend_from_slice(&val.to_le_bytes());
        }
        if self.v2 {
            out.extend_from_slice(&(if self.pairwise { 1u32 } else { 0u32 }).to_le_bytes());
        }
        let mut push_slice = |s: &[f32]| {
            for &x in s {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        push_slice(&self.ft);
        push_slice(&self.ftb);
        push_slice(&self.psqt);
        push_slice(&self.l1_w);
        push_slice(&self.l1_b);
        push_slice(&self.l2_w);
        push_slice(&self.l2_b);
        push_slice(&self.out_w);
        push_slice(&self.out_b);
        if self.v2 {
            push_slice(&self.s57w);
        }
        out
    }

    #[inline]
    pub fn din(&self) -> usize {
        if self.pairwise {
            self.width
        } else {
            2 * self.width
        }
    }

    #[inline]
    pub fn out_in(&self) -> usize {
        if self.v2 {
            self.l1 + self.l2
        } else {
            self.l2
        }
    }

    /// Forward + backward on a single position, accumulating gradients into `grad`.
    /// Returns the sample loss `|sigmoid(y) - target_prob|^2.5`.
    pub fn forward_backward_sample(
        &self,
        b: &Board,
        target_prob: f32,
        grad: &mut BatchGradients,
    ) -> f32 {
        let (us, nu, them, nt) = nnfeat::both(b, self.groups);
        let w = self.width;
        let l1 = self.l1;
        let l2 = self.l2;
        let din = self.din();
        let out_in = self.out_in();
        let empties = b.empty().count_ones() as usize;
        let bucket = (empties * self.buckets / 50).min(self.buckets - 1);

        // 1. Feature transformer forward
        let mut raw_acc = vec![0f32; 2 * w];
        let mut clipped_acc = vec![0f32; 2 * w];
        let mut ps = [0f32; 2];

        for (half, (idx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            let a = &mut raw_acc[half * w..(half + 1) * w];
            a.copy_from_slice(&self.ftb);
            for &f in &idx[..n] {
                let fi = f as usize;
                let row = &self.ft[fi * w..(fi + 1) * w];
                for (x, r) in a.iter_mut().zip(row) {
                    *x += *r;
                }
                ps[half] += self.psqt[fi * self.buckets + bucket];
            }
            let c = &mut clipped_acc[half * w..(half + 1) * w];
            for j in 0..w {
                c[j] = a[j].clamp(0.0, 1.0);
            }
        }

        // 2. Pairwise or CReLU concat
        let mut x_in = vec![0f32; din];
        if self.pairwise {
            let h = w / 2;
            for half in 0..2 {
                let c = &clipped_acc[half * w..(half + 1) * w];
                for j in 0..h {
                    x_in[half * h + j] = c[j] * c[h + j];
                }
            }
        } else {
            x_in.copy_from_slice(&clipped_acc);
        }

        // 3. L1 forward
        let mut raw_h1 = vec![0f32; l1];
        let mut h1 = vec![0f32; l1];
        for j in 0..l1 {
            let r = bucket * l1 + j;
            let row = &self.l1_w[r * din..(r + 1) * din];
            let s: f32 = row.iter().zip(&x_in).map(|(a, b)| a * b).sum();
            let v = s + self.l1_b[r];
            raw_h1[j] = v;
            h1[j] = v.clamp(0.0, 1.0);
        }

        // 4. L2 forward
        let mut raw_h2 = vec![0f32; l2];
        let mut h2 = vec![0f32; l2];
        for j in 0..l2 {
            let r = bucket * l2 + j;
            let row = &self.l2_w[r * l1..(r + 1) * l1];
            let s: f32 = row.iter().zip(&h1).map(|(a, b)| a * b).sum();
            let v = s + self.l2_b[r];
            raw_h2[j] = v;
            h2[j] = v.clamp(0.0, 1.0);
        }

        // 5. Output forward
        let out_row = &self.out_w[bucket * out_in..(bucket + 1) * out_in];
        let o = if self.v2 {
            let s1: f32 = out_row[..l1].iter().zip(&h1).map(|(a, b)| a * b).sum();
            let s2: f32 = out_row[l1..].iter().zip(&h2).map(|(a, b)| a * b).sum();
            s1 + s2 + self.out_b[bucket]
        } else {
            out_row.iter().zip(&h2).map(|(a, b)| a * b).sum::<f32>() + self.out_b[bucket]
        };
        let y = o + 0.5 * (ps[0] - ps[1]);

        // 6. Loss = |sigmoid(y) - target_prob|^2.5
        let pred = 1.0 / (1.0 + (-y).exp());
        let diff = pred - target_prob;
        let abs_diff = diff.abs();
        let loss = abs_diff.powf(2.5);
        if abs_diff < 1e-7 {
            return loss;
        }
        let d_loss_d_pred = 2.5 * diff.signum() * abs_diff.powf(1.5);
        let dy = d_loss_d_pred * pred * (1.0 - pred);

        // 7. Backward: Output layer
        grad.out_b[bucket] += dy;
        let mut dh1 = vec![0f32; l1];
        let mut dh2 = vec![0f32; l2];
        let out_grad_row = &mut grad.out_w[bucket * out_in..(bucket + 1) * out_in];
        if self.v2 {
            for j in 0..l1 {
                out_grad_row[j] += dy * h1[j];
                dh1[j] += dy * out_row[j];
            }
            for j in 0..l2 {
                out_grad_row[l1 + j] += dy * h2[j];
                dh2[j] += dy * out_row[l1 + j];
            }
        } else {
            for j in 0..l2 {
                out_grad_row[j] += dy * h2[j];
                dh2[j] += dy * out_row[j];
            }
        }

        // 8. Backward: L2
        for j in 0..l2 {
            if raw_h2[j] <= 0.0 || raw_h2[j] >= 1.0 {
                continue;
            }
            let g = dh2[j];
            let r = bucket * l2 + j;
            grad.l2_b[r] += g;
            let w_row = &self.l2_w[r * l1..(r + 1) * l1];
            let g_row = &mut grad.l2_w[r * l1..(r + 1) * l1];
            for i in 0..l1 {
                g_row[i] += g * h1[i];
                dh1[i] += g * w_row[i];
            }
        }

        // 9. Backward: L1
        let mut dx = vec![0f32; din];
        for j in 0..l1 {
            if raw_h1[j] <= 0.0 || raw_h1[j] >= 1.0 {
                continue;
            }
            let g = dh1[j];
            let r = bucket * l1 + j;
            grad.l1_b[r] += g;
            let w_row = &self.l1_w[r * din..(r + 1) * din];
            let g_row = &mut grad.l1_w[r * din..(r + 1) * din];
            for i in 0..din {
                g_row[i] += g * x_in[i];
                dx[i] += g * w_row[i];
            }
        }

        // 10. Backward: Pairwise / CReLU -> Feature Transformer
        let mut d_acc = vec![0f32; 2 * w];
        if self.pairwise {
            let h = w / 2;
            for half in 0..2 {
                let c = &clipped_acc[half * w..(half + 1) * w];
                let r_a = &raw_acc[half * w..(half + 1) * w];
                let d_a = &mut d_acc[half * w..(half + 1) * w];
                for j in 0..h {
                    let gx = dx[half * h + j];
                    if r_a[j] > 0.0 && r_a[j] < 1.0 {
                        d_a[j] = gx * c[h + j];
                    }
                    if r_a[h + j] > 0.0 && r_a[h + j] < 1.0 {
                        d_a[h + j] = gx * c[j];
                    }
                }
            }
        } else {
            for i in 0..(2 * w) {
                if raw_acc[i] > 0.0 && raw_acc[i] < 1.0 {
                    d_acc[i] = dx[i];
                }
            }
        }

        // 11. Sparse feature & PSQT accumulation (only touches active features!)
        for (half, (idx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            let da = &d_acc[half * w..(half + 1) * w];
            for j in 0..w {
                grad.ftb[j] += da[j];
            }
            let ps_sign = if half == 0 { 0.5 * dy } else { -0.5 * dy };
            for &f in &idx[..n] {
                let fi = f as usize;
                grad.touched_feats.insert(fi as u16);
                let ft_g = &mut grad.ft[fi * w..(fi + 1) * w];
                for j in 0..w {
                    ft_g[j] += da[j];
                }
                grad.psqt[fi * self.buckets + bucket] += ps_sign;
            }
        }

        loss
    }
}

pub struct BatchGradients {
    pub ft: Vec<f32>,
    pub ftb: Vec<f32>,
    pub psqt: Vec<f32>,
    pub l1_w: Vec<f32>,
    pub l1_b: Vec<f32>,
    pub l2_w: Vec<f32>,
    pub l2_b: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    pub touched_feats: HashSet<u16>,
}

impl BatchGradients {
    pub fn new(net: &TrainableNet) -> Self {
        BatchGradients {
            ft: vec![0.0; net.ft.len()],
            ftb: vec![0.0; net.ftb.len()],
            psqt: vec![0.0; net.psqt.len()],
            l1_w: vec![0.0; net.l1_w.len()],
            l1_b: vec![0.0; net.l1_b.len()],
            l2_w: vec![0.0; net.l2_w.len()],
            l2_b: vec![0.0; net.l2_b.len()],
            out_w: vec![0.0; net.out_w.len()],
            out_b: vec![0.0; net.out_b.len()],
            touched_feats: HashSet::with_capacity(512),
        }
    }

    pub fn merge_from(&mut self, other: &BatchGradients, width: usize, buckets: usize) {
        for (a, b) in self.ftb.iter_mut().zip(&other.ftb) {
            *a += *b;
        }
        for (a, b) in self.l1_w.iter_mut().zip(&other.l1_w) {
            *a += *b;
        }
        for (a, b) in self.l1_b.iter_mut().zip(&other.l1_b) {
            *a += *b;
        }
        for (a, b) in self.l2_w.iter_mut().zip(&other.l2_w) {
            *a += *b;
        }
        for (a, b) in self.l2_b.iter_mut().zip(&other.l2_b) {
            *a += *b;
        }
        for (a, b) in self.out_w.iter_mut().zip(&other.out_w) {
            *a += *b;
        }
        for (a, b) in self.out_b.iter_mut().zip(&other.out_b) {
            *a += *b;
        }
        for &f in &other.touched_feats {
            let fi = f as usize;
            self.touched_feats.insert(f);
            let dst_ft = &mut self.ft[fi * width..(fi + 1) * width];
            let src_ft = &other.ft[fi * width..(fi + 1) * width];
            for (a, b) in dst_ft.iter_mut().zip(src_ft) {
                *a += *b;
            }
            let dst_ps = &mut self.psqt[fi * buckets..(fi + 1) * buckets];
            let src_ps = &other.psqt[fi * buckets..(fi + 1) * buckets];
            for (a, b) in dst_ps.iter_mut().zip(src_ps) {
                *a += *b;
            }
        }
    }

    pub fn clear_sparse(&mut self, width: usize, buckets: usize) {
        self.ftb.fill(0.0);
        self.l1_w.fill(0.0);
        self.l1_b.fill(0.0);
        self.l2_w.fill(0.0);
        self.l2_b.fill(0.0);
        self.out_w.fill(0.0);
        self.out_b.fill(0.0);
        for &f in &self.touched_feats {
            let fi = f as usize;
            self.ft[fi * width..(fi + 1) * width].fill(0.0);
            self.psqt[fi * buckets..(fi + 1) * buckets].fill(0.0);
        }
        self.touched_feats.clear();
    }
}

pub struct SparseAdamW {
    step: u32,
    m_ft: Vec<f32>,
    v_ft: Vec<f32>,
    m_ftb: Vec<f32>,
    v_ftb: Vec<f32>,
    m_psqt: Vec<f32>,
    v_psqt: Vec<f32>,
    m_l1_w: Vec<f32>,
    v_l1_w: Vec<f32>,
    m_l1_b: Vec<f32>,
    v_l1_b: Vec<f32>,
    m_l2_w: Vec<f32>,
    v_l2_w: Vec<f32>,
    m_l2_b: Vec<f32>,
    v_l2_b: Vec<f32>,
    m_out_w: Vec<f32>,
    v_out_w: Vec<f32>,
    m_out_b: Vec<f32>,
    v_out_b: Vec<f32>,
}

impl SparseAdamW {
    pub fn new(net: &TrainableNet) -> Self {
        SparseAdamW {
            step: 0,
            m_ft: vec![0.0; net.ft.len()],
            v_ft: vec![0.0; net.ft.len()],
            m_ftb: vec![0.0; net.ftb.len()],
            v_ftb: vec![0.0; net.ftb.len()],
            m_psqt: vec![0.0; net.psqt.len()],
            v_psqt: vec![0.0; net.psqt.len()],
            m_l1_w: vec![0.0; net.l1_w.len()],
            v_l1_w: vec![0.0; net.l1_w.len()],
            m_l1_b: vec![0.0; net.l1_b.len()],
            v_l1_b: vec![0.0; net.l1_b.len()],
            m_l2_w: vec![0.0; net.l2_w.len()],
            v_l2_w: vec![0.0; net.l2_w.len()],
            m_l2_b: vec![0.0; net.l2_b.len()],
            v_l2_b: vec![0.0; net.l2_b.len()],
            m_out_w: vec![0.0; net.out_w.len()],
            v_out_w: vec![0.0; net.out_w.len()],
            m_out_b: vec![0.0; net.out_b.len()],
            v_out_b: vec![0.0; net.out_b.len()],
        }
    }

    pub fn step_batch(
        &mut self,
        net: &mut TrainableNet,
        grad: &BatchGradients,
        batch_size: usize,
        lr: f32,
        wd: f32,
    ) {
        self.step += 1;
        let inv_n = 1.0 / (batch_size.max(1) as f32);
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let corr1 = 1.0 - b1.powi(self.step as i32);
        let corr2 = 1.0 - b2.powi(self.step as i32);
        let step_lr = lr / corr1;

        let update_slice = |w: &mut [f32],
                            m: &mut [f32],
                            v: &mut [f32],
                            g: &[f32],
                            clamp_qa: bool| {
            for i in 0..w.len() {
                let gi = g[i] * inv_n;
                m[i] = b1 * m[i] + (1.0 - b1) * gi;
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                let denom = (v[i] / corr2).sqrt() + 1e-8;
                w[i] = w[i] * (1.0 - lr * wd) - step_lr * (m[i] / denom);
                if clamp_qa {
                    w[i] = w[i].clamp(-QA_CLAMP, QA_CLAMP);
                }
            }
        };

        update_slice(&mut net.ftb, &mut self.m_ftb, &mut self.v_ftb, &grad.ftb, true);
        update_slice(&mut net.l1_w, &mut self.m_l1_w, &mut self.v_l1_w, &grad.l1_w, false);
        update_slice(&mut net.l1_b, &mut self.m_l1_b, &mut self.v_l1_b, &grad.l1_b, false);
        update_slice(&mut net.l2_w, &mut self.m_l2_w, &mut self.v_l2_w, &grad.l2_w, false);
        update_slice(&mut net.l2_b, &mut self.m_l2_b, &mut self.v_l2_b, &grad.l2_b, false);
        update_slice(&mut net.out_w, &mut self.m_out_w, &mut self.v_out_w, &grad.out_w, false);
        update_slice(&mut net.out_b, &mut self.m_out_b, &mut self.v_out_b, &grad.out_b, false);

        // Sparse update: ONLY touch active feature rows in `ft` and `psqt`!
        let width = net.width;
        let buckets = net.buckets;
        for &f in &grad.touched_feats {
            let fi = f as usize;
            let ft_range = fi * width..(fi + 1) * width;
            update_slice(
                &mut net.ft[ft_range.clone()],
                &mut self.m_ft[ft_range.clone()],
                &mut self.v_ft[ft_range.clone()],
                &grad.ft[ft_range],
                true,
            );
            let ps_range = fi * buckets..(fi + 1) * buckets;
            update_slice(
                &mut net.psqt[ps_range.clone()],
                &mut self.m_psqt[ps_range.clone()],
                &mut self.v_psqt[ps_range.clone()],
                &grad.psqt[ps_range],
                false,
            );
        }
    }
}

/// Load labeled samples from `dis_positions.bin` + completed `c_<cid>.txt` chunks in `labels_dir`.
pub fn load_relabel_dataset(
    bin_path: &str,
    labels_dir: &str,
    max_chunks: Option<usize>,
) -> Result<Vec<LabeledSample>, String> {
    let bin_bytes = fs::read(bin_path).map_err(|e| format!("read {bin_path}: {e}"))?;
    let total_rows = bin_bytes.len() / 25;
    let total_chunks = (total_rows + 4999) / 5000;
    let limit = max_chunks.unwrap_or(total_chunks).min(total_chunks);
    let mut samples = Vec::new();

    for cid in 0..limit {
        let p_c = format!("{labels_dir}/c_{cid}.txt");
        let p_chunk = format!("{labels_dir}/chunk_{cid:05}.txt");
        let path = if Path::new(&p_c).exists() {
            p_c
        } else if Path::new(&p_chunk).exists() {
            p_chunk
        } else {
            continue;
        };
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let lo = cid * 5000;
        let hi = ((cid + 1) * 5000).min(total_rows);
        let want = hi.saturating_sub(lo);
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() != want {
            continue;
        }
        for (i, line) in lines.into_iter().enumerate() {
            let Some(score_str) = line.split_whitespace().next() else {
                continue;
            };
            let Ok(cp) = score_str.parse::<i32>() else {
                continue;
            };
            let row_idx = lo + i;
            let rec = &bin_bytes[row_idx * 25..row_idx * 25 + 25];
            let u = |k: usize| u64::from_le_bytes(rec[k..k + 8].try_into().unwrap());
            let clamped_cp = (cp as f32).clamp(-1500.0, 1500.0);
            let target_prob = 1.0 / (1.0 + (-clamped_cp / 400.0).exp());
            samples.push(LabeledSample {
                occ0: u(0),
                occ1: u(8),
                blk: u(16),
                stm: rec[24],
                target_prob,
            });
        }
    }
    Ok(samples)
}

pub fn run_train_cli(
    init_spec: &str,
    bin_path: &str,
    labels_dir: &str,
    out_path: &str,
    epochs: usize,
    batch_size: usize,
    lr: f32,
    wd: f32,
    threads: usize,
    max_chunks: Option<usize>,
) -> ExitCode {
    println!(
        "=== TitaniumTest Bullet-Style Sparse NNUE Trainer ===\n\
         Init Net   : {init_spec}\n\
         Dataset    : {bin_path} + {labels_dir}\n\
         Output     : {out_path}\n\
         Config     : epochs={epochs}, batch_size={batch_size}, lr={lr}, wd={wd}, threads={threads}"
    );

    let init_bytes = if init_spec == "bundled" {
        BUNDLED_NNUE.to_vec()
    } else {
        match fs::read(init_spec) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("train: cannot read {init_spec}: {e}");
                return ExitCode::FAILURE;
            }
        }
    };

    let mut net = match TrainableNet::from_tnn_bytes(&init_bytes) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("train: parse init net failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let t_load = Instant::now();
    let samples = match load_relabel_dataset(bin_path, labels_dir, max_chunks) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("train: load dataset failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if samples.is_empty() {
        eprintln!("train: 0 valid labeled samples found in {labels_dir}");
        return ExitCode::FAILURE;
    }
    println!(
        "[Train] Loaded {} labeled positions ({:.1} MB in RAM, 0 disk dumps) in {:.1}s",
        samples.len(),
        (samples.len() * std::mem::size_of::<LabeledSample>()) as f64 / 1_048_576.0,
        t_load.elapsed().as_secs_f64()
    );

    let mut opt = SparseAdamW::new(&net);
    let mut master_grad = BatchGradients::new(&net);
    let mut rng_state = 0x853C_49E6_748F_EA9Bu64;
    let mut indices: Vec<usize> = (0..samples.len()).collect();

    for ep in 1..=epochs {
        let t_ep = Instant::now();
        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }

        // Cosine LR decay across epochs
        let progress = (ep - 1) as f32 / (epochs.max(1) as f32);
        let cur_lr = lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()).max(0.05);

        let mut total_loss = 0.0f64;
        let mut total_seen = 0usize;

        for batch_indices in indices.chunks(batch_size) {
            master_grad.clear_sparse(net.width, net.buckets);
            let chunk_per_thread = (batch_indices.len() + threads - 1) / threads;
            let partial_loss = Mutex::new(0.0f64);
            let master_mutex = Mutex::new(&mut master_grad);

            std::thread::scope(|sc| {
                for sub in batch_indices.chunks(chunk_per_thread.max(1)) {
                    let net_ref = &net;
                    let samples_ref = &samples;
                    let partial_loss = &partial_loss;
                    let master_mutex = &master_mutex;
                    sc.spawn(move || {
                        let mut local_grad = BatchGradients::new(net_ref);
                        let mut loc_loss = 0.0f64;
                        for &idx in sub {
                            let s = &samples_ref[idx];
                            let b = Board::from_occ(s.occ0, s.occ1, s.blk, s.stm);
                            loc_loss += net_ref.forward_backward_sample(&b, s.target_prob, &mut local_grad) as f64;
                        }
                        *partial_loss.lock().unwrap() += loc_loss;
                        master_mutex
                            .lock()
                            .unwrap()
                            .merge_from(&local_grad, net_ref.width, net_ref.buckets);
                    });
                }
            });

            total_loss += *partial_loss.lock().unwrap();
            total_seen += batch_indices.len();
            opt.step_batch(&mut net, &master_grad, batch_indices.len(), cur_lr, wd);
        }

        let mean_loss = total_loss / (total_seen.max(1) as f64);
        let dt = t_ep.elapsed().as_secs_f64().max(1e-6);
        let pos_per_sec = total_seen as f64 / dt;
        println!(
            "[Epoch {ep:>2}/{epochs}] loss={mean_loss:.6} | lr={cur_lr:.6} | speed={pos_per_sec:.0} pos/s ({dt:.1}s)"
        );

        let bytes = net.to_tnn_bytes();
        if let Some(parent) = Path::new(out_path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let mut f = fs::File::create(out_path).expect("create out tnn");
        f.write_all(&bytes).expect("write out tnn");
    }

    println!("[Train] Saved quantized-ready TNN network to {out_path}");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use titanium::nnue::Nnue;

    #[test]
    fn bundled_net_roundtrips_and_trains_cleanly() {
        let mut tnet = TrainableNet::from_tnn_bytes(BUNDLED_NNUE).expect("parse bundled");
        let exported = tnet.to_tnn_bytes();
        assert_eq!(exported.len(), BUNDLED_NNUE.len());

        let b = Board::start();
        let ref_nnue = Nnue::bundled();
        let ref_y = ref_nnue.forward_f32(&b, 0.0);

        let mut grad = BatchGradients::new(&tnet);
        let target_p = 0.85f32;
        let loss0 = tnet.forward_backward_sample(&b, target_p, &mut grad);
        let expected_loss = ((1.0 / (1.0 + (-ref_y).exp())) - target_p).abs().powf(2.5);
        assert!(
            (loss0 - expected_loss).abs() < 1e-5,
            "forward parity mismatch: {loss0} vs {expected_loss}"
        );

        let mut opt = SparseAdamW::new(&tnet);
        opt.step_batch(&mut tnet, &grad, 1, 1e-3, 0.0);
        grad.clear_sparse(tnet.width, tnet.buckets);
        let loss1 = tnet.forward_backward_sample(&b, target_p, &mut grad);
        assert!(
            loss1 < loss0,
            "loss did not decrease after SparseAdamW step: {loss1} >= {loss0}"
        );
    }
}
