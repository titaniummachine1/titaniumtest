//! `lab ws-probe`: Weight-Space Programming probe & adjuster on trained `TNN2`/`TNN3` nets (e.g. `nn_v13anneal`, champion).
//!
//! 1. `--inspect NET.tnn`: inspect current `psqt` weights on G1 safe stones (`empty_neighbours == 0`, corner/edge/interior)
//!    and G3 solid 3x3 block centers (`own_nb == 8, enemy_nb == 0`) across the 16 phase buckets.
//! 2. `--fit NET.tnn --train TRAIN.tdat --val VAL.tdat [--n-train 400000] --out-prefix STEM`:
//!    Strictly using training rows (`--train`), solves per-bucket IRLS (`|sigmoid(y)-t|^2.5`, `[0.55, 0.55, 0.25]` rw targets)
//!    for zero-cost linear adjustments in weight space ONLY:
//!      - `_safe.tnn`: adjusts ONLY `psqt` on G1 safe stones (corner/edge/interior, symmetric STM - NTM)
//!      - `_safecal.tnn`: also applies per-bucket affine scale + bias (`out_w`, `psqt`, `out_b`)
//!    Reloads modified nets with `Nnue::load`, verifies held-out `--val` MSE (first 131k Modal slice + full 1.53M tail with paired SE/z).
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::process::ExitCode;

use titanium::nnfeat::{
    both as feat_both, G1_OFF, G2_NONE, G2_OFF, G3_OFF, G5_OFF, G6_OFF, G6_PAR_OFF, NUM_FEATS, SQ_STATES,
};
use titanium::nnue::Nnue;
use titanium::Board;

use crate::hcetrain::{sig, target};
use crate::util::{arg, Rec, REC};

const FULL: u64 = (1u64 << 49) - 1;
const FILE_A: u64 = {
    let mut m = 0u64;
    let mut r = 0;
    while r < 7 {
        m |= 1u64 << (r * 7);
        r += 1;
    }
    m
};
const FILE_G: u64 = FILE_A << 6;
const CORNERS: u64 = (1u64 << 0) | (1u64 << 6) | (1u64 << 42) | (1u64 << 48);
const INTERIOR_5X5: u64 = {
    let mut m = 0u64;
    let mut r = 1;
    while r <= 5 {
        let mut c = 1;
        while c <= 5 {
            m |= 1u64 << (r * 7 + c);
            c += 1;
        }
        r += 1;
    }
    m
};

#[derive(Clone)]
struct RawTnn {
    magic: [u8; 4],
    num_feats: usize,
    width: usize,
    l1: usize,
    l2: usize,
    buckets: usize,
    groups: u32,
    pairwise: bool,
    ft: Vec<f32>,
    ftb: Vec<f32>,
    psqt: Vec<f32>,
    l1_w: Vec<f32>,
    l1_b: Vec<f32>,
    l2_w: Vec<f32>,
    l2_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    s57w: Vec<f32>,
    dis_w: Vec<f32>,
    dis_b: Vec<f32>,
}

impl RawTnn {
    fn load(path: &str) -> Result<Self, String> {
        let d = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let magic: [u8; 4] = d[..4].try_into().map_err(|_| "short header")?;
        let v2 = &magic == b"TNN2";
        let dis = &magic == b"TNN3";
        if d.len() < 32 || (!v2 && !dis && &magic != b"TNN1") {
            return Err("not a TNN1/TNN2/TNN3 file".into());
        }
        let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        let (num_feats, width, l1, l2, buckets, groups) = (u(0), u(1), u(2), u(3), u(4), u(5) as u32);
        let pairwise = (v2 || dis) && u(6) != 0;
        let din = if pairwise { width } else { 2 * width };
        let out_in = if v2 { l1 + l2 } else { l2 };
        let mut p = if v2 || dis { 32 } else { 28 };
        let mut take = |n: usize| -> Vec<f32> {
            let v = d[p..p + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            p += 4 * n;
            v
        };
        let out = RawTnn {
            magic,
            num_feats,
            width,
            l1,
            l2,
            buckets,
            groups,
            pairwise,
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
            dis_w: if dis { take(buckets * (l1 + l2)) } else { Vec::new() },
            dis_b: if dis { take(buckets) } else { Vec::new() },
        };
        if p != d.len() {
            return Err(format!("trailing/missing bytes ({p} vs {})", d.len()));
        }
        Ok(out)
    }

    fn save(&self, path: &str) -> Result<(), String> {
        let mut out = Vec::with_capacity(self.ft.len() * 4 + 65536);
        out.extend_from_slice(&self.magic);
        for &val in &[self.num_feats as u32, self.width as u32, self.l1 as u32, self.l2 as u32, self.buckets as u32, self.groups] {
            out.extend_from_slice(&val.to_le_bytes());
        }
        if &self.magic == b"TNN2" || &self.magic == b"TNN3" {
            out.extend_from_slice(&(if self.pairwise { 1u32 } else { 0u32 }).to_le_bytes());
        }
        let mut push = |s: &[f32]| {
            for &x in s {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        push(&self.ft);
        push(&self.ftb);
        push(&self.psqt);
        push(&self.l1_w);
        push(&self.l1_b);
        push(&self.l2_w);
        push(&self.l2_b);
        push(&self.out_w);
        push(&self.out_b);
        if &self.magic == b"TNN2" {
            push(&self.s57w);
        }
        if &self.magic == b"TNN3" {
            push(&self.dis_w);
            push(&self.dis_b);
        }
        if let Some(par) = std::path::Path::new(path).parent() {
            let _ = fs::create_dir_all(par);
        }
        fs::write(path, out).map_err(|e| format!("{path}: {e}"))
    }

    fn bucket(&self, empties: usize) -> usize {
        let b = (empties * self.buckets) / 50;
        b.min(self.buckets - 1)
    }
}

fn board_of(r: &Rec) -> Board {
    let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
    Board::from_occ(own, opp, 0, 0)
}

fn safe_mask(stones: u64, empty: u64) -> u64 {
    let h = empty | ((empty & !FILE_A) >> 1) | ((empty & !FILE_G) << 1);
    let near_e = (h | (h >> 7) | (h << 7)) & FULL;
    stones & !near_e
}

fn inspect_net(path: &str) {
    let raw = RawTnn::load(path).expect("load raw tnn");
    println!("=== psqt inspection: {path} (buckets={}, map_id={}) ===", raw.buckets, raw.groups >> 8);
    println!("  NOTE: total STM-NTM effect of symmetric (psqt[own] += d, psqt[enemy] -= d) is 400*d cp per safe-stone diff");
    for bk in [0, 1, 2, 3, 4, 6, 8, 12, 15].into_iter().filter(|&b| b < raw.buckets) {
        let mut sum = [0f64; 6];
        let mut cnt = [0usize; 6];
        for sq in 0..49usize {
            let bit = 1u64 << sq;
            let o_safe = raw.psqt[(G1_OFF + sq * SQ_STATES) * raw.buckets + bk] as f64;
            let e_safe = raw.psqt[(G1_OFF + sq * SQ_STATES + 18) * raw.buckets + bk] as f64;
            let net_safe_cp = (o_safe - e_safe) * 200.0; // 0.5 * 400 * (o - e)
            let cls = if (CORNERS & bit) != 0 { 0 } else if (INTERIOR_5X5 & bit) == 0 { 1 } else { 2 };
            sum[cls] += net_safe_cp;
            cnt[cls] += 1;
            // non-safe stones: mean of empty_nb 1..8, both unexposed & exposed
            for code in 1..18usize {
                if code == 9 {
                    continue;
                }
                let o_u = raw.psqt[(G1_OFF + sq * SQ_STATES + code) * raw.buckets + bk] as f64;
                let e_u = raw.psqt[(G1_OFF + sq * SQ_STATES + 18 + code) * raw.buckets + bk] as f64;
                sum[3] += (o_u - e_u) * 200.0;
                cnt[3] += 1;
            }
            if (INTERIOR_5X5 & bit) != 0 {
                let g3_o = raw.psqt[(G3_OFF + ((sq * 2) * 9 + 8) * 9) * raw.buckets + bk] as f64;
                let g3_e = raw.psqt[(G3_OFF + ((sq * 2 + 1) * 9 + 8) * 9) * raw.buckets + bk] as f64;
                sum[4] += (g3_o - g3_e) * 200.0;
                cnt[4] += 1;
            }
        }
        println!(
            "  bk {:>2}: safe_cp corner {:+6.1} | edge {:+6.1} | int {:+6.1} | nonsafe_avg {:+6.1} | g3_solid3 {:+6.1}",
            bk,
            sum[0] / cnt[0] as f64,
            sum[1] / cnt[1] as f64,
            sum[2] / cnt[2] as f64,
            sum[3] / cnt[3] as f64,
            sum[4] / cnt[4].max(1) as f64,
        );
    }
}

fn d4_orbit_id(sq: usize) -> usize {
    let (r, c) = (sq / 7, sq % 7);
    let r = r.min(6 - r);
    let c = c.min(6 - c);
    let (a, b) = if r <= c { (r, c) } else { (c, r) };
    // (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,1)=4, (1,2)=5, (1,3)=6, (2,2)=7, (2,3)=8, (3,3)=9
    match (a, b) {
        (0, 0) => 0,
        (0, 1) => 1,
        (0, 2) => 2,
        (0, 3) => 3,
        (1, 1) => 4,
        (1, 2) => 5,
        (1, 3) => 6,
        (2, 2) => 7,
        (2, 3) => 8,
        _ => 9,
    }
}

fn feature_group_stats(raw: &RawTnn, name: &str, indices: &[usize]) {
    let w = raw.width;
    let nb = raw.buckets;
    let mut ft_zero = 0usize;
    let mut ft_weak = 0usize;
    let mut ps_zero = 0usize;
    let mut ft_sum_rms = 0.0f64;
    let mut ft_max_rms = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut clamped = 0usize;
    let mut ps_sum_rms = 0.0f64;
    for &f in indices {
        if f >= raw.num_feats {
            continue;
        }
        let row = &raw.ft[f * w..(f + 1) * w];
        let mut s2 = 0.0f64;
        for &v in row {
            let av = v.abs();
            if av > max_abs {
                max_abs = av;
            }
            if av >= 1.97 {
                clamped += 1;
            }
            s2 += (v as f64) * (v as f64);
        }
        let l2 = s2.sqrt();
        let rms = (s2 / w as f64).sqrt();
        if l2 < 1e-4 {
            ft_zero += 1;
        } else if l2 < 1e-2 {
            ft_weak += 1;
        }
        ft_sum_rms += rms;
        if rms > ft_max_rms {
            ft_max_rms = rms;
        }
        let prow = &raw.psqt[f * nb..(f + 1) * nb];
        let ps2: f64 = prow.iter().map(|&v| (v as f64) * (v as f64)).sum();
        if ps2.sqrt() < 1e-4 {
            ps_zero += 1;
        }
        ps_sum_rms += (ps2 / nb as f64).sqrt();
    }
    let n = indices.len().max(1) as f64;
    println!(
        "  {:<12} n={:<5} | ft_zero {:>5} ({:>5.1}%) tiny<0.01 {:>5} | ft_rms mean {:.4} max {:.4} | |w|max {:.3} clamp>=1.97 {:>4} | psqt_zero {:>5} rms {:.4}",
        name,
        indices.len(),
        ft_zero,
        100.0 * ft_zero as f64 / n,
        ft_weak,
        ft_sum_rms / n,
        ft_max_rms,
        max_abs,
        clamped,
        ps_zero,
        ps_sum_rms / n,
    );
}

fn d4_symmetry_report(raw: &RawTnn, feats_by_sq_code: &[usize], codes: usize, label: &str) {
    // feats_by_sq_code has length 49 * codes indexed sq * codes + code
    let w = raw.width;
    let nb = raw.buckets;
    let mut ft_var_sum = 0.0f64;
    let mut ft_sig_sum = 0.0f64;
    let mut ps_var_sum = 0.0f64;
    let mut ps_sig_sum = 0.0f64;
    let mut n_orb = 0usize;
    for code in 0..codes {
        let mut orbs = [Vec::<usize>::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for sq in 0..49usize {
            orbs[d4_orbit_id(sq)].push(feats_by_sq_code[sq * codes + code]);
        }
        for orb in &orbs {
            if orb.len() < 2 {
                continue;
            }
            let k = orb.len() as f64;
            let mut active = false;
            for j in 0..w {
                let mut m = 0.0f64;
                for &f in orb {
                    m += raw.ft[f * w + j] as f64;
                }
                m /= k;
                let mut v = 0.0f64;
                for &f in orb {
                    let d = raw.ft[f * w + j] as f64 - m;
                    v += d * d;
                }
                if m * m + v / k > 1e-8 {
                    active = true;
                }
                ft_var_sum += v / k;
                ft_sig_sum += m * m;
            }
            for bk in 0..nb {
                let mut m = 0.0f64;
                for &f in orb {
                    m += raw.psqt[f * nb + bk] as f64;
                }
                m /= k;
                let mut v = 0.0f64;
                for &f in orb {
                    let d = raw.psqt[f * nb + bk] as f64 - m;
                    v += d * d;
                }
                ps_var_sum += v / k;
                ps_sig_sum += m * m;
            }
            if active {
                n_orb += 1;
            }
        }
    }
    let ft_ratio = (ft_var_sum / ft_sig_sum.max(1e-12)).sqrt();
    let ps_ratio = (ps_var_sum / ps_sig_sum.max(1e-12)).sqrt();
    let ps_noise_cp = (ps_var_sum / (n_orb.max(1) * nb) as f64).sqrt() * 200.0;
    println!(
        "  D4 symmetry {:<10}: ft_asym/mean {:.1}% | psqt_asym/mean {:.1}% (avg within-orbit std {:.2} cp)",
        label,
        100.0 * ft_ratio,
        100.0 * ps_ratio,
        ps_noise_cp
    );
}

fn tensor_health(name: &str, data: &[f32]) {
    if data.is_empty() {
        return;
    }
    let mut nans = 0usize;
    let mut zero = 0usize;
    let mut clamped = 0usize;
    let mut s2 = 0.0f64;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &x in data {
        if !x.is_finite() {
            nans += 1;
            continue;
        }
        if x.abs() < 1e-5 {
            zero += 1;
        }
        if x.abs() >= 1.97 {
            clamped += 1;
        }
        min_v = min_v.min(x);
        max_v = max_v.max(x);
        s2 += (x as f64) * (x as f64);
    }
    let rms = (s2 / data.len() as f64).sqrt();
    println!(
        "  {:<8} n={:<8} | rms {:.4} [{:+.3}, {:+.3}] | |w|<1e-5 {:>6} ({:.2}%) | |w|>=1.97 {:>4} | nan {}",
        name,
        data.len(),
        rms,
        min_v,
        max_v,
        zero,
        100.0 * zero as f64 / data.len() as f64,
        clamped,
        nans
    );
}

fn audit_net(path: &str, val_path: Option<&str>, n_val: usize) {
    let raw = RawTnn::load(path).expect("load raw tnn");
    let magic_s = std::str::from_utf8(&raw.magic).unwrap_or("????");
    let v2 = &raw.magic == b"TNN2";
    let din = if raw.pairwise { raw.width } else { 2 * raw.width };
    let out_in = if v2 { raw.l1 + raw.l2 } else { raw.l2 };
    let total_params = raw.ft.len()
        + raw.ftb.len()
        + raw.psqt.len()
        + raw.l1_w.len()
        + raw.l1_b.len()
        + raw.l2_w.len()
        + raw.l2_b.len()
        + raw.out_w.len()
        + raw.out_b.len()
        + raw.s57w.len()
        + raw.dis_w.len()
        + raw.dis_b.len();
    println!(
        "=== FULL WEIGHT AUDIT: {path} ===\n  magic={magic_s} feats={} width={} (din={}) l1={} l2={} buckets={} pairwise={} groups={:#x} (map_id={}) | params={} ({:.2} MB)",
        raw.num_feats,
        raw.width,
        din,
        raw.l1,
        raw.l2,
        raw.buckets,
        raw.pairwise,
        raw.groups,
        raw.groups >> 8,
        total_params,
        (total_params * 4) as f64 / (1024.0 * 1024.0)
    );

    println!("\n[1] Global Tensor Health:");
    tensor_health("ft", &raw.ft);
    tensor_health("ftb", &raw.ftb);
    tensor_health("psqt", &raw.psqt);
    tensor_health("l1_w", &raw.l1_w);
    tensor_health("l1_b", &raw.l1_b);
    tensor_health("l2_w", &raw.l2_w);
    tensor_health("l2_b", &raw.l2_b);
    tensor_health("out_w", &raw.out_w);
    tensor_health("out_b", &raw.out_b);
    tensor_health("s57w", &raw.s57w);
    tensor_health("dis_w", &raw.dis_w);
    tensor_health("dis_b", &raw.dis_b);

    let i16_risk = raw.ft.iter().filter(|&&x| (x * 1024.0).abs() >= 24000.0).count();
    let i16_max = raw.ft.iter().map(|&x| (x * 1024.0).abs()).fold(0.0f32, f32::max);
    println!("  ft int16-quant risk (|w*1024| >= 24000): {i16_risk} weights (max |w*1024| = {i16_max:.1} / 32767)");

    println!("\n[2] Feature-Group Breakdown (rows = features):");
    let mut g1_st = Vec::new();
    let mut g1_emp = Vec::new();
    let mut g1_blk = Vec::new();
    for sq in 0..49usize {
        for c in 0..36usize {
            g1_st.push(G1_OFF + sq * SQ_STATES + c);
        }
        for c in 36..261usize {
            g1_emp.push(G1_OFF + sq * SQ_STATES + c);
        }
        g1_blk.push(G1_OFF + sq * SQ_STATES + 261);
    }
    let g2_rel: Vec<usize> = (G2_OFF..G2_NONE).collect();
    let g2_none = vec![G2_NONE];
    let g3_all: Vec<usize> = (G3_OFF..G5_OFF).collect();
    let g5_all: Vec<usize> = (G5_OFF..G6_OFF).collect();
    let g6_all: Vec<usize> = (G6_OFF..G6_PAR_OFF).collect();
    let g6_par: Vec<usize> = (G6_PAR_OFF..NUM_FEATS).collect();
    feature_group_stats(&raw, "G1_stones", &g1_st);
    feature_group_stats(&raw, "G1_empties", &g1_emp);
    feature_group_stats(&raw, "G1_blocker", &g1_blk);
    feature_group_stats(&raw, "G2_anchor", &g2_rel);
    feature_group_stats(&raw, "G2_none", &g2_none);
    feature_group_stats(&raw, "G3_ring", &g3_all);
    feature_group_stats(&raw, "G5_2ply", &g5_all);
    feature_group_stats(&raw, "G6_cnt", &g6_all);
    feature_group_stats(&raw, "G6_parity", &g6_par);

    println!("\n[3] FT Column Health & D4 Symmetry Dispersion:");
    let w = raw.width;
    let mut col_rms = vec![0.0f64; w];
    for f in 0..raw.num_feats {
        let row = &raw.ft[f * w..(f + 1) * w];
        for j in 0..w {
            col_rms[j] += (row[j] as f64) * (row[j] as f64);
        }
    }
    for j in 0..w {
        col_rms[j] = (col_rms[j] / raw.num_feats as f64).sqrt();
    }
    let dead_cols: Vec<usize> = (0..w).filter(|&j| col_rms[j] < 0.01).collect();
    let mut sorted_rms = col_rms.clone();
    sorted_rms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  FT col RMS: min {:.4}, p10 {:.4}, median {:.4}, max {:.4} | dead cols (<0.01): {:?}",
        sorted_rms[0],
        sorted_rms[w / 10],
        sorted_rms[w / 2],
        sorted_rms[w - 1],
        dead_cols
    );
    // Max cosine similarity between FT columns
    let mut col_norms = vec![0.0f64; w];
    for j in 0..w {
        let mut s = 0.0f64;
        for f in 0..raw.num_feats {
            let v = raw.ft[f * w + j] as f64;
            s += v * v;
        }
        col_norms[j] = s.sqrt().max(1e-12);
    }
    let mut best_cos = (0.0f64, 0usize, 0usize);
    let mut gt80 = 0usize;
    for i in 0..w {
        for j in (i + 1)..w {
            let mut dot = 0.0f64;
            for f in 0..raw.num_feats {
                dot += (raw.ft[f * w + i] as f64) * (raw.ft[f * w + j] as f64);
            }
            let cos = dot.abs() / (col_norms[i] * col_norms[j]);
            if cos > 0.80 {
                gt80 += 1;
            }
            if cos > best_cos.0 {
                best_cos = (cos, i, j);
            }
        }
    }
    println!(
        "  FT column redundancy: max |cos|={:.4} (cols {},{}) | pairs |cos|>0.80: {}",
        best_cos.0, best_cos.1, best_cos.2, gt80
    );

    d4_symmetry_report(&raw, &g1_st, 36, "G1_stones");
    d4_symmetry_report(&raw, &g5_all, 9, "G5_2ply");
    // For G3, sq * (2*81) + code
    let mut g3_by_sq = Vec::with_capacity(49 * 162);
    for sq in 0..49usize {
        for c in 0..162usize {
            g3_by_sq.push(G3_OFF + sq * 162 + c);
        }
    }
    d4_symmetry_report(&raw, &g3_by_sq, 162, "G3_ring");

    let Some(vp) = val_path else {
        println!("\n(pass --val VAL.tdat to run dynamic activation & feature hit audit)");
        return;
    };

    println!("\n[4] Loading {n_val} strided validation positions from {vp}...");
    let rows = read_strided_train(vp, n_val);
    let net_ref = Nnue::load(path).expect("load reference nnue");
    let nb = raw.buckets;
    let l1 = raw.l1;
    let l2 = raw.l2;

    let mut feat_hits = vec![0u32; raw.num_feats];
    let mut ft_zero_cnt = vec![0u64; 2 * w];
    let mut ft_one_cnt = vec![0u64; 2 * w];
    let mut ft_sum = vec![0.0f64; 2 * w];
    let mut ft_sum2 = vec![0.0f64; 2 * w];
    let mut pw_nz = vec![0u64; w];

    let mut bk_n = vec![0u64; nb];
    let mut bk_w = vec![0.0f64; nb];
    let mut bk_mse = vec![0.0f64; nb];
    let mut bk_err_ps = vec![0.0f64; nb];
    let mut bk_err2 = vec![0.0f64; nb];
    let mut bk_ps2 = vec![0.0f64; nb];
    let mut bk_mlp2 = vec![0.0f64; nb];

    let mut h1_zero = vec![vec![0u64; l1]; nb];
    let mut h1_one = vec![vec![0u64; l1]; nb];
    let mut h1_sum = vec![vec![0.0f64; l1]; nb];
    let mut h1_sum2 = vec![vec![0.0f64; l1]; nb];

    let mut h2_zero = vec![vec![0u64; l2]; nb];
    let mut h2_one = vec![vec![0u64; l2]; nb];
    let mut h2_sum = vec![vec![0.0f64; l2]; nb];
    let mut h2_sum2 = vec![vec![0.0f64; l2]; nb];
    let mut h2_dot = vec![vec![0.0f64; l2 * l2]; nb];

    let mut acc = vec![0f32; 2 * w];
    let mut x = vec![0f32; din];
    let mut h1 = vec![0f32; l1];
    let mut h2 = vec![0f32; l2];
    let mut max_parity_diff = 0.0f32;

    for (idx, r) in rows.iter().enumerate() {
        let b = board_of(r);
        let empties = b.empty().count_ones() as usize;
        let bk = raw.bucket(empties);
        let (us, nu, them, nt) = feat_both(&b, raw.groups);
        let mut ps = [0f32; 2];
        for (half, (fidx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            let off = half * w;
            acc[off..off + w].copy_from_slice(&raw.ftb);
            for &f in &fidx[..n] {
                let fi = f as usize;
                feat_hits[fi] += 1;
                let row = &raw.ft[fi * w..(fi + 1) * w];
                for j in 0..w {
                    acc[off + j] += row[j];
                }
                ps[half] += raw.psqt[fi * nb + bk];
            }
            for j in 0..w {
                let raw_v = acc[off + j];
                if raw_v <= 0.0 {
                    ft_zero_cnt[off + j] += 1;
                }
                if raw_v >= 1.0 {
                    ft_one_cnt[off + j] += 1;
                }
                let cv = raw_v.clamp(0.0, 1.0);
                acc[off + j] = cv;
                ft_sum[off + j] += cv as f64;
                ft_sum2[off + j] += (cv as f64) * (cv as f64);
            }
        }
        if raw.pairwise {
            let h = w / 2;
            for half in 0..2 {
                let a = &acc[half * w..(half + 1) * w];
                for j in 0..h {
                    let prod = a[j] * a[h + j];
                    x[half * h + j] = prod;
                    if prod > 0.0 {
                        pw_nz[half * h + j] += 1;
                    }
                }
            }
        } else {
            x.copy_from_slice(&acc);
        }
        for j in 0..l1 {
            let r_idx = bk * l1 + j;
            let row = &raw.l1_w[r_idx * din..(r_idx + 1) * din];
            let s: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() + raw.l1_b[r_idx];
            if s <= 0.0 {
                h1_zero[bk][j] += 1;
            }
            if s >= 1.0 {
                h1_one[bk][j] += 1;
            }
            let v = s.clamp(0.0, 1.0);
            h1[j] = v;
            h1_sum[bk][j] += v as f64;
            h1_sum2[bk][j] += (v as f64) * (v as f64);
        }
        for k in 0..l2 {
            let r_idx = bk * l2 + k;
            let row = &raw.l2_w[r_idx * l1..(r_idx + 1) * l1];
            let s: f32 = row.iter().zip(&h1).map(|(a, b)| a * b).sum::<f32>() + raw.l2_b[r_idx];
            if s <= 0.0 {
                h2_zero[bk][k] += 1;
            }
            if s >= 1.0 {
                h2_one[bk][k] += 1;
            }
            let v = s.clamp(0.0, 1.0);
            h2[k] = v;
            h2_sum[bk][k] += v as f64;
            h2_sum2[bk][k] += (v as f64) * (v as f64);
        }
        for a in 0..l2 {
            let va = h2[a] as f64;
            for c in a..l2 {
                h2_dot[bk][a * l2 + c] += va * (h2[c] as f64);
            }
        }
        let orow = &raw.out_w[bk * out_in..(bk + 1) * out_in];
        let mlp = if v2 {
            let s1: f32 = orow[..l1].iter().zip(&h1).map(|(a, b)| a * b).sum();
            let s2: f32 = orow[l1..].iter().zip(&h2).map(|(a, b)| a * b).sum();
            s1 + s2 + raw.out_b[bk]
        } else {
            orow.iter().zip(&h2).map(|(a, b)| a * b).sum::<f32>() + raw.out_b[bk]
        };
        let psqt_v = 0.5 * (ps[0] - ps[1]);
        let y = mlp + psqt_v;
        if idx < 200 {
            let d = (y - net_ref.forward(&b, 0.0)).abs();
            if d > max_parity_diff {
                max_parity_diff = d;
            }
        }
        let p = sig(y) as f64;
        let t = target(r, true) as f64;
        let wt = r.wsum.max(0.0) as f64;
        let err = t - p;
        bk_n[bk] += 1;
        bk_w[bk] += wt;
        bk_mse[bk] += wt * err * err;
        bk_err_ps[bk] += wt * err * (psqt_v as f64);
        bk_err2[bk] += wt * err * err;
        bk_ps2[bk] += wt * (psqt_v as f64) * (psqt_v as f64);
        bk_mlp2[bk] += wt * (mlp as f64) * (mlp as f64);
    }

    println!("  forward parity check vs Nnue::forward (first 200 pos): max |diff| = {max_parity_diff:.6}");

    // Check active feature hits vs weight norms
    let active_feats = feat_hits.iter().filter(|&&c| c > 0).count();
    let hit_ge10 = feat_hits.iter().filter(|&&c| c >= 10).count();
    let mut visited_weak = Vec::new();
    for (f, &cnt) in feat_hits.iter().enumerate() {
        if cnt >= 20 {
            let row = &raw.ft[f * w..(f + 1) * w];
            let l2n: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
            if l2n < 1e-2 {
                visited_weak.push((f, cnt, l2n));
            }
        }
    }
    println!(
        "  Feature coverage on {} samples: {}/{} hit (>0), {} hit (>=10) | frequent (>=20) with ft_L2<0.01: {}",
        rows.len(),
        active_feats,
        raw.num_feats,
        hit_ge10,
        visited_weak.len()
    );

    // Check FT neuron activity
    let nf = (rows.len() * 2) as f64; // wait: STM tracked in 0..w, NTM in w..2w
    let n_pos = rows.len() as f64;
    let (mut ft_dead, mut ft_sat, mut ft_low_std) = (0usize, 0usize, 0usize);
    let mut ft_act_min = 1.0f64;
    let mut ft_act_max = 0.0f64;
    for i in 0..(2 * w) {
        let act = 1.0 - (ft_zero_cnt[i] as f64 / n_pos);
        let one = ft_one_cnt[i] as f64 / n_pos;
        let m = ft_sum[i] / n_pos;
        let v = (ft_sum2[i] / n_pos - m * m).max(0.0).sqrt();
        ft_act_min = ft_act_min.min(act);
        ft_act_max = ft_act_max.max(act);
        if act < 0.005 {
            ft_dead += 1;
        }
        if one > 0.90 {
            ft_sat += 1;
        }
        if v < 0.01 {
            ft_low_std += 1;
        }
    }
    let _ = nf;
    println!(
        "  FT CReLU neurons (2x{w} = {}): active frac min {:.1}% max {:.1}% | dead(<0.5% active) {} | sat(>90% ==1) {} | low_std(<0.01) {}",
        2 * w,
        100.0 * ft_act_min,
        100.0 * ft_act_max,
        ft_dead,
        ft_sat,
        ft_low_std
    );
    if raw.pairwise {
        let dead_pw = pw_nz.iter().filter(|&&c| (c as f64 / n_pos) < 0.002).count();
        println!("  Pairwise inputs ({w}): active <0.2%: {dead_pw}");
    }

    println!("\n[5] Per-Bucket PSQT vs MLP & Hidden Neuron Audit (L1={l1}, L2={l2}):");
    let mut total_dead_l1 = 0usize;
    let mut total_sat_l1 = 0usize;
    let mut total_weak_l1 = 0usize;
    let mut total_dead_l2 = 0usize;
    let mut total_sat_l2 = 0usize;
    let mut total_weak_l2 = 0usize;
    let mut dup_l2_pairs = 0usize;

    for bk in 0..nb {
        let n = bk_n[bk] as f64;
        if n < 1.0 {
            continue;
        }
        let wsum = bk_w[bk].max(1e-12);
        let mse = bk_mse[bk] / wsum;
        let ps_rms_cp = (bk_ps2[bk] / wsum).sqrt() * 400.0;
        let mlp_rms_cp = (bk_mlp2[bk] / wsum).sqrt() * 400.0;
        let r_ep = bk_err_ps[bk] / (bk_err2[bk] * bk_ps2[bk]).sqrt().max(1e-12);

        let mut dead1 = Vec::new();
        let mut sat1 = Vec::new();
        let mut weak1 = Vec::new();
        for j in 0..l1 {
            let act = 1.0 - (h1_zero[bk][j] as f64 / n);
            let one = h1_one[bk][j] as f64 / n;
            let m = h1_sum[bk][j] / n;
            let std = (h1_sum2[bk][j] / n - m * m).max(0.0).sqrt();
            let l2_col_norm: f64 = (0..l2)
                .map(|k| {
                    let v = raw.l2_w[(bk * l2 + k) * l1 + j] as f64;
                    v * v
                })
                .sum::<f64>()
                .sqrt();
            let direct_out = if v2 { raw.out_w[bk * out_in + j].abs() as f64 } else { 0.0 };
            let gain = l2_col_norm + direct_out;
            if act < 0.01 || std < 1e-3 {
                dead1.push((j, act, one, std));
            } else if one > 0.85 {
                sat1.push((j, one));
            } else if std * gain < 0.005 {
                weak1.push((j, std * gain));
            }
        }

        let mut dead2 = Vec::new();
        let mut sat2 = Vec::new();
        let mut weak2 = Vec::new();
        let mut h2_stds = vec![0.0f64; l2];
        let mut h2_means = vec![0.0f64; l2];
        for k in 0..l2 {
            let act = 1.0 - (h2_zero[bk][k] as f64 / n);
            let one = h2_one[bk][k] as f64 / n;
            let m = h2_sum[bk][k] / n;
            let std = (h2_sum2[bk][k] / n - m * m).max(0.0).sqrt();
            h2_means[k] = m;
            h2_stds[k] = std;
            let ow_idx = bk * out_in + if v2 { l1 + k } else { k };
            let cp_rms = 400.0 * (raw.out_w[ow_idx].abs() as f64) * std;
            if act < 0.01 || std < 1e-3 {
                dead2.push((k, act, one, cp_rms));
            } else if one > 0.85 {
                sat2.push((k, one));
            } else if cp_rms < 0.25 {
                weak2.push((k, cp_rms));
            }
        }
        let mut bk_dups = 0usize;
        for a in 0..l2 {
            for c in (a + 1)..l2 {
                if h2_stds[a] < 1e-3 || h2_stds[c] < 1e-3 {
                    continue;
                }
                let cov = h2_dot[bk][a * l2 + c] / n - h2_means[a] * h2_means[c];
                let corr = cov / (h2_stds[a] * h2_stds[c]);
                if corr.abs() > 0.95 {
                    bk_dups += 1;
                }
            }
        }

        total_dead_l1 += dead1.len();
        total_sat_l1 += sat1.len();
        total_weak_l1 += weak1.len();
        total_dead_l2 += dead2.len();
        total_sat_l2 += sat2.len();
        total_weak_l2 += weak2.len();
        dup_l2_pairs += bk_dups;

        println!(
            "  bk {:>2} (n={:>5}): mse {:.5} | psqt_rms {:5.1} cp, mlp_rms {:5.1} cp, r(err,psqt) {:+.3} | L1 dead={} sat={} weak={} | L2 dead={} sat={} weak={} dup={}",
            bk,
            bk_n[bk],
            mse,
            ps_rms_cp,
            mlp_rms_cp,
            r_ep,
            dead1.len(),
            sat1.len(),
            weak1.len(),
            dead2.len(),
            sat2.len(),
            weak2.len(),
            bk_dups
        );
        if !dead1.is_empty() {
            println!("    dead L1 ids: {:?}", dead1.iter().map(|x| x.0).collect::<Vec<_>>());
        }
        if !dead2.is_empty() {
            println!("    dead L2 ids: {:?}", dead2.iter().map(|x| x.0).collect::<Vec<_>>());
        }
    }
    println!(
        "\n  TOTALS across {} buckets: L1 ({} neurons): dead={}, sat={}, weak={} | L2 ({} neurons): dead={}, sat={}, weak={}, corr>0.95 pairs={}",
        nb,
        nb * l1,
        total_dead_l1,
        total_sat_l1,
        total_weak_l1,
        nb * l2,
        total_dead_l2,
        total_sat_l2,
        total_weak_l2,
        dup_l2_pairs
    );
}

fn vec_cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt().max(1e-12)
}

fn decode_feat(f: usize) -> String {
    if f < G2_OFF {
        let sq = f / SQ_STATES;
        let code = f % SQ_STATES;
        let (r, c) = (sq / 7, sq % 7);
        if code < 36 {
            let side = if code < 18 { "own" } else { "enemy" };
            let lc = code % 18;
            let exp = lc / 9;
            let nb = lc % 9;
            format!("G1({r},{c}) {side} exp={exp} empty_nb={nb}")
        } else if code < 261 {
            format!("G1_empty({r},{c}) code={code}")
        } else {
            format!("G1_blk({r},{c})")
        }
    } else if f < G3_OFF {
        format!("G2({})", f - G2_OFF)
    } else if f < G5_OFF {
        let z = f - G3_OFF;
        let sc = z / 81;
        let sq = sc / 2;
        let col = if sc % 2 == 0 { "own" } else { "enemy" };
        let rc = z % 81;
        format!("G3({},{}) {col} own_nb={} foe_nb={}", sq / 7, sq % 7, rc / 9, rc % 9)
    } else if f < G6_OFF {
        let z = f - G5_OFF;
        let sq = z / 9;
        let dv = (z % 9) as i32 - 3;
        format!("G5({},{}) 2ply={dv:+}", sq / 7, sq % 7)
    } else if f < G6_PAR_OFF {
        format!("G6_cnt({})", f - G6_OFF)
    } else {
        format!("G6_par({})", f - G6_PAR_OFF)
    }
}

fn deep_net_report(path: &str, val_path: &str, n_val: usize, d4_out: Option<&str>) {
    let mut raw = RawTnn::load(path).expect("load raw tnn");
    let nb = raw.buckets;
    let w = raw.width;
    let l1 = raw.l1;
    let l2 = raw.l2;
    let v2 = &raw.magic == b"TNN2";
    let din = if raw.pairwise { w } else { 2 * w };
    let out_in = if v2 { l1 + l2 } else { l2 };

    println!("=== DEEP NEURON / BUCKET / FEATURE DIAGNOSTIC: {path} ===");
    println!("\n[A] Adjacent Bucket Cosine Similarity (check 8->16 bucket cloning):");
    for bk in (0..nb - 1).step_by(2) {
        let c_l1 = vec_cos(&raw.l1_w[bk * l1 * din..(bk + 1) * l1 * din], &raw.l1_w[(bk + 1) * l1 * din..(bk + 2) * l1 * din]);
        let c_l2 = vec_cos(&raw.l2_w[bk * l2 * l1..(bk + 1) * l2 * l1], &raw.l2_w[(bk + 1) * l2 * l1..(bk + 2) * l2 * l1]);
        let c_out = vec_cos(&raw.out_w[bk * out_in..(bk + 1) * out_in], &raw.out_w[(bk + 1) * out_in..(bk + 2) * out_in]);
        println!("  bk {:>2} vs {:>2} | cos(l1_w) {:+.4} | cos(l2_w) {:+.4} | cos(out_w) {:+.4}", bk, bk + 1, c_l1, c_l2, c_out);
    }
    println!("  bk 14 vs 15 | cos(out_w) {:+.4} | out_b[14]={:+.4}, out_b[15]={:+.4}",
        vec_cos(&raw.out_w[14 * out_in..15 * out_in], &raw.out_w[15 * out_in..16 * out_in]),
        raw.out_b[14], raw.out_b[15]
    );
    let bk15_l1_zero = (0..l1).filter(|&j| {
        let s: f64 = raw.l1_w[(15 * l1 + j) * din..(15 * l1 + j + 1) * din].iter().map(|&x| (x as f64)*(x as f64)).sum();
        s < 1e-6
    }).count();
    let bk15_l2_zero = (0..l2).filter(|&k| {
        let s: f64 = raw.l2_w[(15 * l2 + k) * l1..(15 * l2 + k + 1) * l1].iter().map(|&x| (x as f64)*(x as f64)).sum();
        s < 1e-6
    }).count();
    let bk15_l2_neg_b = (0..l2).filter(|&k| raw.l2_b[15 * l2 + k] < -0.5).count();
    println!("  bk 15 static check: zero_l1={bk15_l1_zero}/{l1}, zero_l2={bk15_l2_zero}/{l2}, strong_neg_l2_b(<-0.5)={bk15_l2_neg_b}/{l2}");

    let rows = read_strided_train(val_path, n_val);
    let mut feat_hits = vec![0u32; raw.num_feats];
    let mut h1_min = vec![vec![f32::INFINITY; l1]; nb];
    let mut h1_max = vec![vec![f32::NEG_INFINITY; l1]; nb];
    let mut h1_zero = vec![vec![0u64; l1]; nb];
    let mut h1_one = vec![vec![0u64; l1]; nb];
    let mut h2_min = vec![vec![f32::INFINITY; l2]; nb];
    let mut h2_max = vec![vec![f32::NEG_INFINITY; l2]; nb];
    let mut h2_zero = vec![vec![0u64; l2]; nb];
    let mut h2_one = vec![vec![0u64; l2]; nb];
    let mut bk_n = vec![0u64; nb];

    let mut acc = vec![0f32; 2 * w];
    let mut x = vec![0f32; din];
    let mut h1 = vec![0f32; l1];

    for r in &rows {
        let b = board_of(r);
        let bk = raw.bucket(b.empty().count_ones() as usize);
        bk_n[bk] += 1;
        let (us, nu, them, nt) = feat_both(&b, raw.groups);
        for (half, (fidx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            let off = half * w;
            acc[off..off + w].copy_from_slice(&raw.ftb);
            for &f in &fidx[..n] {
                let fi = f as usize;
                feat_hits[fi] += 1;
                let row = &raw.ft[fi * w..(fi + 1) * w];
                for j in 0..w {
                    acc[off + j] += row[j];
                }
            }
            for j in 0..w {
                acc[off + j] = acc[off + j].clamp(0.0, 1.0);
            }
        }
        if raw.pairwise {
            let h = w / 2;
            for half in 0..2 {
                let a = &acc[half * w..(half + 1) * w];
                for j in 0..h {
                    x[half * h + j] = a[j] * a[h + j];
                }
            }
        } else {
            x.copy_from_slice(&acc);
        }
        for j in 0..l1 {
            let r_idx = bk * l1 + j;
            let row = &raw.l1_w[r_idx * din..(r_idx + 1) * din];
            let s: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() + raw.l1_b[r_idx];
            h1_min[bk][j] = h1_min[bk][j].min(s);
            h1_max[bk][j] = h1_max[bk][j].max(s);
            if s <= 0.0 { h1_zero[bk][j] += 1; }
            if s >= 1.0 { h1_one[bk][j] += 1; }
            h1[j] = s.clamp(0.0, 1.0);
        }
        for k in 0..l2 {
            let r_idx = bk * l2 + k;
            let row = &raw.l2_w[r_idx * l1..(r_idx + 1) * l1];
            let s: f32 = row.iter().zip(&h1).map(|(a, b)| a * b).sum::<f32>() + raw.l2_b[r_idx];
            h2_min[bk][k] = h2_min[bk][k].min(s);
            h2_max[bk][k] = h2_max[bk][k].max(s);
            if s <= 0.0 { h2_zero[bk][k] += 1; }
            if s >= 1.0 { h2_one[bk][k] += 1; }
        }
    }

    println!("\n[B] All Dead/Saturated Hidden Neurons Details:");
    for bk in 0..nb {
        if bk_n[bk] == 0 { continue; }
        let n = bk_n[bk] as f64;
        for j in 0..l1 {
            let act = 1.0 - h1_zero[bk][j] as f64 / n;
            let one = h1_one[bk][j] as f64 / n;
            if act < 0.01 || one > 0.85 {
                let wn: f64 = raw.l1_w[(bk * l1 + j) * din..(bk * l1 + j + 1) * din].iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>().sqrt();
                let coln: f64 = (0..l2).map(|k| { let v = raw.l2_w[(bk * l2 + k) * l1 + j] as f64; v*v }).sum::<f64>().sqrt();
                println!(
                    "  L1 bk {:>2} n={:>2} | act={:>5.1}% one={:>5.1}% | pre_act [{:+.3}, {:+.3}] | b={:+.4} |w|_2={:.3} out_col_L2={:.3}",
                    bk, j, 100.0*act, 100.0*one, h1_min[bk][j], h1_max[bk][j], raw.l1_b[bk*l1+j], wn, coln
                );
            }
        }
        for k in 0..l2 {
            let act = 1.0 - h2_zero[bk][k] as f64 / n;
            let one = h2_one[bk][k] as f64 / n;
            if act < 0.01 || one > 0.85 {
                let wn: f64 = raw.l2_w[(bk * l2 + k) * l1..(bk * l2 + k + 1) * l1].iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>().sqrt();
                let ow = raw.out_w[bk * out_in + if v2 { l1 + k } else { k }];
                println!(
                    "  L2 bk {:>2} n={:>2} | act={:>5.1}% one={:>5.1}% | pre_act [{:+.3}, {:+.3}] | b={:+.4} |w|_2={:.3} out_w={:+.4}",
                    bk, k, 100.0*act, 100.0*one, h2_min[bk][k], h2_max[bk][k], raw.l2_b[bk*l2+k], wn, ow
                );
            }
        }
    }

    println!("\n[C] Frequently-Hit Features (hits>=100 in {n_val}) with Undertrained FT (<0.0145) or Near-Zero PSQT:");
    let mut low_ft = Vec::new();
    let mut zero_psqt = Vec::new();
    for (f, &cnt) in feat_hits.iter().enumerate() {
        if cnt < 100 { continue; }
        let ft_rms = (raw.ft[f*w..(f+1)*w].iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>() / w as f64).sqrt();
        let ps_rms = (raw.psqt[f*nb..(f+1)*nb].iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>() / nb as f64).sqrt();
        if ft_rms < 0.0145 {
            low_ft.push((cnt, ft_rms, ps_rms, f));
        }
        if ps_rms < 1e-4 {
            zero_psqt.push((cnt, ft_rms, f));
        }
    }
    low_ft.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    zero_psqt.sort_by(|a, b| b.0.cmp(&a.0));
    println!("  Visited >=100 hits with ft_rms < 0.0145: {} | with zero psqt: {}", low_ft.len(), zero_psqt.len());
    for (cnt, ft_r, ps_r, f) in low_ft.iter().take(12) {
        println!("    undertrained FT: hits={cnt:>6} ft_rms={ft_r:.4} psqt_rms={ps_r:.4} | {}", decode_feat(*f));
    }
    for (cnt, ft_r, f) in zero_psqt.iter().take(12) {
        println!("    zero PSQT      : hits={cnt:>6} ft_rms={ft_r:.4} | {}", decode_feat(*f));
    }

    println!("\n[D] Top D4 PSQT Symmetry Outliers (within-orbit spread in cp, bk 8 [~26 empties]):");
    let mut outliers = Vec::new();
    let check_set = |name: &str, base: usize, stride: usize, codes: usize, outliers: &mut Vec<(f64, String)>| {
        for code in 0..codes {
            let mut orbs = [Vec::<usize>::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for sq in 0..49usize {
                orbs[d4_orbit_id(sq)].push(base + sq * stride + code);
            }
            for (oid, orb) in orbs.iter().enumerate() {
                if orb.len() < 2 { continue; }
                let vals: Vec<f64> = orb.iter().map(|&f| raw.psqt[f * nb + 8] as f64 * 200.0).collect();
                let min_v = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let hits: u32 = orb.iter().map(|&f| feat_hits[f]).sum();
                if max_v - min_v > 0.5 && hits >= 20 {
                    outliers.push((max_v - min_v, format!("{name} orbit={oid} code={code} hits={hits} cp=[{:+.1}..{:+.1}] ex={}", min_v, max_v, decode_feat(orb[0]))));
                }
            }
        }
    };
    check_set("G1", G1_OFF, SQ_STATES, 36, &mut outliers);
    check_set("G5", G5_OFF, 9, 9, &mut outliers);
    check_set("G3", G3_OFF, 162, 162, &mut outliers);
    outliers.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (sp, msg) in outliers.iter().take(12) {
        println!("  spread {sp:5.1} cp | {msg}");
    }

    if let Some(out_p) = d4_out {
        println!("\n[E] Testing Weight-Space D4 PSQT Symmetrization + Eval on full {val_path}...");
        let sym_group = |base: usize, stride: usize, codes: usize, r: &mut RawTnn| {
            for code in 0..codes {
                let mut orbs = [Vec::<usize>::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for sq in 0..49usize {
                    orbs[d4_orbit_id(sq)].push(base + sq * stride + code);
                }
                for orb in &orbs {
                    for bk in 0..nb {
                        let mean = orb.iter().map(|&f| r.psqt[f * nb + bk] as f64).sum::<f64>() / orb.len() as f64;
                        for &f in orb {
                            r.psqt[f * nb + bk] = mean as f32;
                        }
                    }
                }
            }
        };
        sym_group(G1_OFF, SQ_STATES, SQ_STATES, &mut raw);
        sym_group(G3_OFF, 162, 162, &mut raw);
        sym_group(G5_OFF, 9, 9, &mut raw);
        raw.save(out_p).expect("save d4 net");
        let full_val = crate::hcetrain::read_rows(val_path, 2_000_000);
        let net_base = Nnue::load(path).expect("reload base");
        let net_d4 = Nnue::load(out_p).expect("reload d4");
        eval_val_report("BASE", &net_base, &full_val);
        eval_val_report("WS_D4_PSQT", &net_d4, &full_val);
        paired_check("BASE", &net_base, "WS_D4_PSQT", &net_d4, &full_val);
    }
}

fn read_strided_train(path: &str, max_rows: usize) -> Vec<Rec> {
    let mut f = File::open(path).expect("open train tdat");
    let total = f.metadata().unwrap().len() as usize / REC;
    let take = max_rows.min(total);
    let stride = (total / take).max(1);
    let mut out = Vec::with_capacity(take);
    let mut buf = [0u8; REC];
    for i in 0..take {
        let pos = ((i * stride) as u64) * REC as u64;
        f.seek(SeekFrom::Start(pos)).unwrap();
        f.read_exact(&mut buf).unwrap();
        out.push(Rec::from_bytes(&buf));
    }
    out
}

/// Gauss elimination for small KxK (K <= 6) linear systems.
fn solve_sym(mut a: [[f64; 6]; 6], mut b: [f64; 6], k: usize) -> [f64; 6] {
    for i in 0..k {
        let mut piv = i;
        for r in (i + 1)..k {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let d = a[i][i];
        if d.abs() < 1e-12 {
            continue;
        }
        for c in i..k {
            a[i][c] /= d;
        }
        b[i] /= d;
        for r in 0..k {
            if r == i {
                continue;
            }
            let f = a[r][i];
            for c in i..k {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    b
}

fn eval_val_report(label: &str, net: &Nnue, val: &[Rec]) {
    let n1 = val.len().min(131_072);
    let mut mse_131_rw = 0.0f64;
    let mut mse_131_of = 0.0f64;
    let mut w1_rw = 0.0f64;
    let mut w1_of = 0.0f64;
    let mut mse_all_of = 0.0f64;
    let mut mse_all_rw = 0.0f64;
    let mut wall_of = 0.0f64;
    let mut wall_rw = 0.0f64;
    let mut r_safe = [0.0f64; 4]; // [sum_xy, sum_xx, sum_yy, n] for 7-49 empties
    for (idx, r) in val.iter().enumerate() {
        let b = board_of(r);
        let y = net.forward(&b, 0.0);
        let p = sig(y) as f64;
        let wt = r.wsum.max(0.0) as f64;
        let t_of = target(r, false) as f64;
        let t_rw = target(r, true) as f64;
        let e_of = (p - t_of) * (p - t_of);
        let e_rw = (p - t_rw) * (p - t_rw);
        if idx < n1 {
            mse_131_of += wt * e_of;
            mse_131_rw += wt * e_rw;
            w1_of += wt;
            w1_rw += wt;
        }
        mse_all_of += wt * e_of;
        mse_all_rw += wt * e_rw;
        wall_of += wt;
        wall_rw += wt;

        let empties = b.empty().count_ones();
        if empties >= 7 {
            let empty = b.empty();
            let sm = safe_mask(b.me(), empty).count_ones() as f64;
            let so = safe_mask(b.opp(), empty).count_ones() as f64;
            let sd = sm - so;
            let err = t_rw - p;
            r_safe[0] += wt * err * sd;
            r_safe[1] += wt * err * err;
            r_safe[2] += wt * sd * sd;
            r_safe[3] += wt;
        }
    }
    let corr = r_safe[0] / (r_safe[1] * r_safe[2]).sqrt().max(1e-12);
    println!(
        "{:<22} | 131k rw {:.6} of {:.6} | full 1.53M rw {:.6} of {:.6} | r(err, safe_diff @7+): {:+.4}",
        label,
        mse_131_rw / w1_rw,
        mse_131_of / w1_of,
        mse_all_rw / wall_rw,
        mse_all_of / wall_of,
        corr
    );
}

fn paired_check(label_a: &str, a: &Nnue, label_b: &str, b: &Nnue, val: &[Rec]) {
    let mut blk_sum = Vec::new();
    let (mut cur_da, mut cur_w) = (0.0f64, 0.0f64);
    let mut n = 0usize;
    for r in val {
        let bd = board_of(r);
        let pa = sig(a.forward(&bd, 0.0)) as f64;
        let pb = sig(b.forward(&bd, 0.0)) as f64;
        let t = target(r, true) as f64;
        let wt = r.wsum.max(0.0) as f64;
        let la = (pa - t).abs().powf(2.5);
        let lb = (pb - t).abs().powf(2.5);
        cur_da += wt * (la - lb);
        cur_w += wt;
        n += 1;
        if n == 2000 {
            blk_sum.push(cur_da / cur_w.max(1e-12));
            cur_da = 0.0;
            cur_w = 0.0;
            n = 0;
        }
    }
    if n > 0 {
        blk_sum.push(cur_da / cur_w.max(1e-12));
    }
    let m = blk_sum.len() as f64;
    let mean = blk_sum.iter().sum::<f64>() / m;
    let var = blk_sum.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (m * (m - 1.0));
    let se = var.sqrt();
    let z = mean / se.max(1e-15);
    println!("  paired pow2.5 ({} vs {}): diff(A-B) {:+.7} +- {:.7} | z {:+.2}", label_a, label_b, mean, se, z);
}

fn apply_safe_deltas(raw: &mut RawTnn, deltas: &[[f64; 3]]) {
    for bk in 0..raw.buckets {
        let [dc, de, di] = deltas[bk];
        for sq in 0..49usize {
            let bit = 1u64 << sq;
            let d = if (CORNERS & bit) != 0 {
                dc
            } else if (INTERIOR_5X5 & bit) == 0 {
                de
            } else {
                di
            } as f32;
            raw.psqt[(G1_OFF + sq * SQ_STATES) * raw.buckets + bk] += d;
            raw.psqt[(G1_OFF + sq * SQ_STATES + 18) * raw.buckets + bk] -= d;
        }
    }
}

fn solve_spd(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let k = b.len();
    for i in 0..k {
        let mut piv = i;
        for r in (i + 1)..k {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let d = a[i][i];
        if d.abs() < 1e-12 {
            continue;
        }
        for c in i..k {
            a[i][c] /= d;
        }
        b[i] /= d;
        for r in 0..k {
            if r == i {
                continue;
            }
            let f = a[r][i];
            if f == 0.0 {
                continue;
            }
            for c in i..k {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    b
}

struct EscapeRec {
    bk: usize,
    t: f32,
    wt: f32,
    ps: f32,
    c_diff: f32,
    e_diff: f32,
    i_diff: f32,
    y0: f32,
    h1: [f32; 16],
    h2: [f32; 32],
}

fn eval_buckets_vs(label_a: &str, a: &Nnue, label_b: &str, b: &Nnue, val: &[Rec], nb: usize) {
    let mut sum_a = vec![0.0f64; nb];
    let mut sum_b = vec![0.0f64; nb];
    let mut w_s = vec![0.0f64; nb];
    for r in val {
        let bd = board_of(r);
        let empties = bd.empty().count_ones() as usize;
        let bk = ((empties * nb) / 50).min(nb - 1);
        let pa = sig(a.forward(&bd, 0.0)) as f64;
        let pb = sig(b.forward(&bd, 0.0)) as f64;
        let t = target(r, true) as f64;
        let wt = r.wsum.max(0.0) as f64;
        sum_a[bk] += wt * (pa - t) * (pa - t);
        sum_b[bk] += wt * (pb - t) * (pb - t);
        w_s[bk] += wt;
    }
    println!("  Per-bucket val rw MSE ({label_a} -> {label_b}):");
    for bk in 0..nb {
        if w_s[bk] < 1.0 {
            continue;
        }
        let ma = sum_a[bk] / w_s[bk];
        let mb = sum_b[bk] / w_s[bk];
        let pct = 100.0 * (mb - ma) / ma;
        println!("    bk {:>2} (w={:>8.0}): {:.6} -> {:.6} ({pct:+.2}%)", bk, w_s[bk], ma, mb);
    }
}

fn escape_fit(net_path: &str, train_path: &str, val_path: &str, n_train: usize, out_path: &str) {
    let mut raw = RawTnn::load(net_path).expect("load raw tnn");
    assert_eq!(raw.l1, 16);
    assert_eq!(raw.l2, 32);
    let nb = raw.buckets;
    let w = raw.width;
    let l1 = raw.l1;
    let l2 = raw.l2;
    let v2 = &raw.magic == b"TNN2";
    let din = if raw.pairwise { w } else { 2 * w };
    let out_in = if v2 { l1 + l2 } else { l2 };

    println!("=== WEIGHT-SPACE LOCAL-MINIMUM ESCAPE FIT: {net_path} ===");
    println!("loading {n_train} strided rows from {train_path} and full val from {val_path}...");
    let train = read_strided_train(train_path, n_train);
    let val = crate::hcetrain::read_rows(val_path, 2_000_000);
    let base_net = Nnue::load(net_path).expect("load base nnue");

    let mut samples = Vec::with_capacity(train.len());
    let mut h2_nz = vec![[0u64; 32]; nb];
    let mut h2_sum = vec![[0.0f64; 32]; nb];
    let mut h2_sum2 = vec![[0.0f64; 32]; nb];
    let mut bk_cnt = vec![0u64; nb];

    let mut acc = vec![0f32; 2 * w];
    let mut x = vec![0f32; din];
    for r in &train {
        let b = board_of(r);
        let empty = b.empty();
        let bk = raw.bucket(empty.count_ones() as usize);
        bk_cnt[bk] += 1;
        let (us, nu, them, nt) = titanium::nnfeat::both(&b, raw.groups);
        let mut ps = [0f32; 2];
        for (half, (fidx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
            let off = half * w;
            acc[off..off + w].copy_from_slice(&raw.ftb);
            for &f in &fidx[..n] {
                let fi = f as usize;
                let row = &raw.ft[fi * w..(fi + 1) * w];
                for j in 0..w {
                    acc[off + j] += row[j];
                }
                ps[half] += raw.psqt[fi * nb + bk];
            }
            for j in 0..w {
                acc[off + j] = acc[off + j].clamp(0.0, 1.0);
            }
        }
        if raw.pairwise {
            let h = w / 2;
            for half in 0..2 {
                let a = &acc[half * w..(half + 1) * w];
                for j in 0..h {
                    x[half * h + j] = a[j] * a[h + j];
                }
            }
        } else {
            x.copy_from_slice(&acc);
        }
        let mut h1 = [0f32; 16];
        for j in 0..16 {
            let r_idx = bk * 16 + j;
            let row = &raw.l1_w[r_idx * din..(r_idx + 1) * din];
            let s: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() + raw.l1_b[r_idx];
            h1[j] = s.clamp(0.0, 1.0);
        }
        let mut h2 = [0f32; 32];
        for k in 0..32 {
            let r_idx = bk * 32 + k;
            let row = &raw.l2_w[r_idx * 16..(r_idx + 1) * 16];
            let s: f32 = row.iter().zip(&h1).map(|(a, b)| a * b).sum::<f32>() + raw.l2_b[r_idx];
            let v = s.clamp(0.0, 1.0);
            h2[k] = v;
            if v > 0.0 {
                h2_nz[bk][k] += 1;
            }
            h2_sum[bk][k] += v as f64;
            h2_sum2[bk][k] += (v as f64) * (v as f64);
        }
        let orow = &raw.out_w[bk * out_in..(bk + 1) * out_in];
        let mlp = if v2 {
            orow[..16].iter().zip(&h1).map(|(a, b)| a * b).sum::<f32>()
                + orow[16..].iter().zip(&h2).map(|(a, b)| a * b).sum::<f32>()
                + raw.out_b[bk]
        } else {
            orow.iter().zip(&h2).map(|(a, b)| a * b).sum::<f32>() + raw.out_b[bk]
        };
        let ps_diff = 0.5 * (ps[0] - ps[1]);
        let y0 = mlp + ps_diff;

        let sm = safe_mask(b.me(), empty);
        let so = safe_mask(b.opp(), empty);
        let c_diff = (sm & CORNERS).count_ones() as i32 - (so & CORNERS).count_ones() as i32;
        let e_diff = (sm & !CORNERS & !INTERIOR_5X5).count_ones() as i32 - (so & !CORNERS & !INTERIOR_5X5).count_ones() as i32;
        let i_diff = (sm & INTERIOR_5X5).count_ones() as i32 - (so & INTERIOR_5X5).count_ones() as i32;

        samples.push(EscapeRec {
            bk,
            t: target(r, true),
            wt: r.wsum.max(0.0),
            ps: ps_diff,
            c_diff: c_diff as f32,
            e_diff: e_diff as f32,
            i_diff: i_diff as f32,
            y0,
            h1,
            h2,
        });
    }

    // Identify dead L2 slots on train samples per bucket, resurrect them via stagewise residual CReLU fit on h1!
    let mut revived_total = 0usize;
    let mut bucket_idxs: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for (idx, s) in samples.iter().enumerate() {
        bucket_idxs[s.bk].push(idx);
    }

    // Store per-sample calibrated prediction (from safecal step) so residual boosting & 37-dim joint IRLS start at safecal
    let mut y_cal_all = vec![0.0f64; samples.len()];
    let mut cal_scales = vec![1.0f64; nb];

    for bk in 0..nb {
        let idxs = &bucket_idxs[bk];
        let n = idxs.len();
        if n < 100 {
            println!("  bk {:>2}: n={n} (<100), skipping", bk);
            continue;
        }
        // Step 1: 5-param safecal baseline [ds, db, dc, de, di] for this bucket
        let mut p = [0.0f64; 5];
        for _iter in 0..6 {
            let mut a = [[0.0f64; 6]; 6];
            let mut rhs = [0.0f64; 6];
            a[0][0] = 500.0;
            for i in 1..5 {
                a[i][i] = 200.0;
            }
            for &si in idxs {
                let s = &samples[si];
                let x = [s.y0 as f64, 1.0, s.c_diff as f64, s.e_diff as f64, s.i_diff as f64];
                let y = s.y0 as f64 + p[0] * x[0] + p[1] + p[2] * x[2] + p[3] * x[3] + p[4] * x[4];
                let sigy = 1.0 / (1.0 + (-y).exp());
                let g = sigy * (1.0 - sigy);
                let err = s.t as f64 - sigy;
                let w_loc = s.wt as f64 * g * g;
                for r_i in 0..5 {
                    rhs[r_i] += s.wt as f64 * g * err * x[r_i];
                    for c_i in 0..5 {
                        a[r_i][c_i] += w_loc * x[r_i] * x[c_i];
                    }
                }
            }
            let dp = solve_sym(a, rhs, 5);
            for i in 0..5 {
                p[i] += dp[i];
            }
        }
        let scale = (1.0 + p[0]) as f32;
        let bias = p[1] as f32;
        cal_scales[bk] = scale as f64;
        for w_v in &mut raw.out_w[bk * out_in..(bk + 1) * out_in] {
            *w_v *= scale;
        }
        raw.out_b[bk] = raw.out_b[bk] * scale + bias;
        for f in 0..raw.num_feats {
            raw.psqt[f * nb + bk] *= scale;
        }
        if !raw.s57w.is_empty() {
            raw.s57w[bk] *= scale;
        }
        for sq in 0..49usize {
            let bit = 1u64 << sq;
            let d = if (CORNERS & bit) != 0 {
                p[2]
            } else if (INTERIOR_5X5 & bit) == 0 {
                p[3]
            } else {
                p[4]
            } as f32;
            raw.psqt[(G1_OFF + sq * SQ_STATES) * nb + bk] += d;
            raw.psqt[(G1_OFF + sq * SQ_STATES + 18) * nb + bk] -= d;
        }

        for &si in idxs {
            let s = &samples[si];
            y_cal_all[si] = s.y0 as f64 + p[0] * s.y0 as f64 + p[1] + p[2] * s.c_diff as f64 + p[3] * s.e_diff as f64 + p[4] * s.i_diff as f64;
        }
    }

    // Recompute bucket residual boosting + full 37-dim head IRLS
    let mut revived_details = Vec::new();
    for bk in 0..nb {
        let idxs = &bucket_idxs[bk];
        let n = idxs.len();
        if n < 100 {
            continue;
        }
        let dead_l2: Vec<usize> = (0..l2)
            .filter(|&k| {
                let act = h2_nz[bk][k] as f64 / n as f64;
                let m = h2_sum[bk][k] / n as f64;
                let sd = (h2_sum2[bk][k] / n as f64 - m * m).max(0.0).sqrt();
                act < 0.01 || sd < 1e-3
            })
            .collect();

        let mut y_cur: Vec<f64> = idxs.iter().map(|&si| y_cal_all[si]).collect();
        let mut boosted_h2: Vec<(usize, Vec<f32>)> = Vec::with_capacity(dead_l2.len());

        // Step 2: Resurrect each dead L2 neuron sequentially via ridge CReLU residual boosting on h1[0..16]
        for &k in &dead_l2 {
            let mut a = vec![vec![0.0f64; 17]; 17];
            let mut rhs = vec![0.0f64; 17];
            for d in 0..16 {
                a[d][d] = 50.0;
            }
            a[16][16] = 20.0;
            for (loc, &si) in idxs.iter().enumerate() {
                let s = &samples[si];
                let sigy = 1.0 / (1.0 + (-y_cur[loc]).exp());
                let g = sigy * (1.0 - sigy);
                let err = s.t as f64 - sigy;
                let w_loc = s.wt as f64 * g * g;
                let gw_err = s.wt as f64 * g * err;
                let mut z = [0.0f64; 17];
                for j in 0..16 {
                    z[j] = s.h1[j] as f64;
                }
                z[16] = 1.0;
                for r_i in 0..17 {
                    rhs[r_i] += gw_err * z[r_i];
                    for c_i in 0..17 {
                        a[r_i][c_i] += w_loc * z[r_i] * z[c_i];
                    }
                }
            }
            let sol = solve_spd(a, rhs);
            // Center & scale pre_act so CReLU lands squarely in active region (mean ~0.40, std ~0.35)
            let mut m = 0.0f64;
            let mut m2 = 0.0f64;
            let mut dot_raw = Vec::with_capacity(n);
            for &si in idxs {
                let s = &samples[si];
                let mut v = 0.0f64;
                for j in 0..16 {
                    v += sol[j] * s.h1[j] as f64;
                }
                dot_raw.push(v);
                m += v;
                m2 += v * v;
            }
            m /= n as f64;
            let sd = (m2 / n as f64 - m * m).max(1e-8).sqrt();
            let gain = 0.35 / sd;
            let mut w_new = [0.0f32; 16];
            for j in 0..16 {
                w_new[j] = (sol[j] * gain).clamp(-1.90, 1.90) as f32;
            }
            let b_new = (0.40 - m * gain).clamp(-1.90, 1.90) as f32;

            let mut acts = Vec::with_capacity(n);
            let mut nz_c = 0usize;
            for (_loc, &si) in idxs.iter().enumerate() {
                let s = &samples[si];
                let mut pa = b_new;
                for j in 0..16 {
                    pa += w_new[j] * s.h1[j];
                }
                let act = pa.clamp(0.0, 1.0);
                if act > 0.0 {
                    nz_c += 1;
                }
                acts.push(act);
            }
            // Optimal 1-D out_w step for residual reduction
            let mut num = 0.0f64;
            let mut den = 400.0f64;
            for (loc, &si) in idxs.iter().enumerate() {
                let s = &samples[si];
                let sigy = 1.0 / (1.0 + (-y_cur[loc]).exp());
                let g = sigy * (1.0 - sigy);
                let err = s.t as f64 - sigy;
                let a_val = acts[loc] as f64;
                num += s.wt as f64 * g * err * a_val;
                den += s.wt as f64 * g * g * a_val * a_val;
            }
            let out_new = (num / den).clamp(-1.90, 1.90);
            for (loc, &a_val) in acts.iter().enumerate() {
                y_cur[loc] += out_new * a_val as f64;
            }
            // Write resurrected neuron into raw.l2_w, raw.l2_b, zero stale out_w (joint step refines all out_w!)
            let r_idx = bk * l2 + k;
            for j in 0..16 {
                raw.l2_w[r_idx * 16 + j] = w_new[j];
            }
            raw.l2_b[r_idx] = b_new;
            let ow_off = bk * out_in + if v2 { l1 + k } else { k };
            raw.out_w[ow_off] = out_new as f32;
            revived_total += 1;
            if revived_details.len() < 12 {
                revived_details.push(format!(
                    "bk {:>2} L2 #{:>2}: act -> {:>5.1}% | b={:+.3} |w|_2={:.3} boost_ow={:+.4}",
                    bk,
                    k,
                    100.0 * nz_c as f64 / n as f64,
                    b_new,
                    w_new.iter().map(|&v| v * v).sum::<f32>().sqrt(),
                    out_new
                ));
            }
            boosted_h2.push((k, acts));
        }

        // Step 3: Joint 37-dim regularized IRLS on [h2[0..32], 1, ps_scaled, c_diff, e_diff, i_diff]
        // Parameterize delta theta around current calibrated+boosted state!
        let mut theta = vec![0.0f64; 37];
        for _iter in 0..5 {
            let mut a = vec![vec![0.0f64; 37]; 37];
            let mut rhs = vec![0.0f64; 37];
            for k in 0..32 {
                a[k][k] = 800.0; // strong L2 prior against overfitting individual out_w weights
            }
            a[32][32] = 400.0; // out_b
            a[33][33] = 800.0; // additional psqt scale
            for d in 34..37 {
                a[d][d] = 400.0; // additional safe deltas
            }

            // Cache current h2_cur per sample in this bucket
            for (loc, &si) in idxs.iter().enumerate() {
                let s = &samples[si];
                let mut f = [0.0f64; 37];
                for k in 0..32 {
                    f[k] = s.h2[k] as f64;
                }
                for (k, acts) in &boosted_h2 {
                    f[*k] = acts[loc] as f64;
                }
                let ps_cal = s.ps as f64 * cal_scales[bk]
                    + 0.0; // note: safe_deltas already in cal y, ps_raw contribution tracked via s.ps
                f[32] = 1.0;
                f[33] = ps_cal;
                f[34] = s.c_diff as f64;
                f[35] = s.e_diff as f64;
                f[36] = s.i_diff as f64;

                let mut y = y_cur[loc];
                for j in 0..37 {
                    y += theta[j] * f[j];
                }
                let sigy = 1.0 / (1.0 + (-y).exp());
                let g = sigy * (1.0 - sigy);
                let err = s.t as f64 - sigy;
                let w_loc = s.wt as f64 * g * g;
                let gw_err = s.wt as f64 * g * err;
                for r_i in 0..37 {
                    rhs[r_i] += gw_err * f[r_i];
                    for c_i in r_i..37 {
                        a[r_i][c_i] += w_loc * f[r_i] * f[c_i];
                    }
                }
            }
            for r_i in 0..37 {
                for c_i in 0..r_i {
                    a[r_i][c_i] = a[c_i][r_i];
                }
            }
            let dp = solve_spd(a, rhs);
            for j in 0..37 {
                theta[j] += dp[j];
            }
        }

        // Apply 37-dim head deltas safely to raw
        for k in 0..32 {
            let ow_off = bk * out_in + if v2 { l1 + k } else { k };
            raw.out_w[ow_off] = (raw.out_w[ow_off] + theta[k] as f32).clamp(-1.95, 1.95);
        }
        raw.out_b[bk] = (raw.out_b[bk] + theta[32] as f32).clamp(-1.95, 1.95);
        let ps_mult = (1.0 + theta[33]) as f32;
        for f in 0..raw.num_feats {
            raw.psqt[f * nb + bk] *= ps_mult;
        }
        for sq in 0..49usize {
            let bit = 1u64 << sq;
            let d = if (CORNERS & bit) != 0 {
                theta[34]
            } else if (INTERIOR_5X5 & bit) == 0 {
                theta[35]
            } else {
                theta[36]
            } as f32;
            raw.psqt[(G1_OFF + sq * SQ_STATES) * nb + bk] += d;
            raw.psqt[(G1_OFF + sq * SQ_STATES + 18) * nb + bk] -= d;
        }
        println!(
            "  bk {:>2} (n={:>5}): dead_L2_revived={} | safecal_scale={:.3} extra_psqt={:.3} | ||d_out||_2={:.3}",
            bk,
            n,
            dead_l2.len(),
            cal_scales[bk],
            ps_mult,
            theta[..32].iter().map(|&v| v * v).sum::<f64>().sqrt()
        );
    }

    println!("\n  Total dead L2 neurons resurrected: {revived_total}");
    for line in &revived_details {
        println!("    {line}");
    }
    raw.save(out_path).expect("save escape net");
    let net_esc = Nnue::load(out_path).expect("reload escape net");
    eval_val_report("BASE", &base_net, &val);
    eval_val_report("WS_ESCAPE", &net_esc, &val);
    paired_check("BASE", &base_net, "WS_ESCAPE", &net_esc, &val);

    let safecal_guess = format!("{}_safecal.tnn", out_path.trim_end_matches("_escape.tnn"));
    if std::path::Path::new(&safecal_guess).exists() {
        let net_sc = Nnue::load(&safecal_guess).expect("load safecal");
        eval_val_report("WS_SAFECAL", &net_sc, &val);
        paired_check("WS_SAFECAL", &net_sc, "WS_ESCAPE", &net_esc, &val);
        eval_buckets_vs("WS_SAFECAL", &net_sc, "WS_ESCAPE", &net_esc, &val, nb);
    } else {
        eval_buckets_vs("BASE", &base_net, "WS_ESCAPE", &net_esc, &val, nb);
    }
}

fn eval_h1_h2(raw: &RawTnn, b: &Board, acc: &mut [f32], x: &mut [f32]) -> (usize, Vec<f32>, Vec<f32>, f32) {
    let w = raw.width;
    let nb = raw.buckets;
    let l1 = raw.l1;
    let l2 = raw.l2;
    let v2 = &raw.magic == b"TNN2";
    let din = x.len();
    let out_in = if v2 { l1 + l2 } else { l2 };
    let bk = raw.bucket(b.empty().count_ones() as usize);
    let (us, nu, them, nt) = feat_both(b, raw.groups);
    let mut ps = [0f32; 2];
    for (half, (fidx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
        let off = half * w;
        acc[off..off + w].copy_from_slice(&raw.ftb);
        for &f in &fidx[..n] {
            let fi = f as usize;
            let row = &raw.ft[fi * w..(fi + 1) * w];
            for j in 0..w {
                acc[off + j] += row[j];
            }
            ps[half] += raw.psqt[fi * nb + bk];
        }
        for j in 0..w {
            acc[off + j] = acc[off + j].clamp(0.0, 1.0);
        }
    }
    if raw.pairwise {
        let h = w / 2;
        for half in 0..2 {
            let a = &acc[half * w..(half + 1) * w];
            for j in 0..h {
                x[half * h + j] = a[j] * a[h + j];
            }
        }
    } else {
        x.copy_from_slice(acc);
    }
    let mut h1 = vec![0f32; l1];
    for j in 0..l1 {
        let r_idx = bk * l1 + j;
        let row = &raw.l1_w[r_idx * din..(r_idx + 1) * din];
        let s: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>() + raw.l1_b[r_idx];
        h1[j] = s.clamp(0.0, 1.0);
    }
    let mut h2 = vec![0f32; l2];
    for k in 0..l2 {
        let r_idx = bk * l2 + k;
        let row = &raw.l2_w[r_idx * l1..(r_idx + 1) * l1];
        let s: f32 = row.iter().zip(h1.iter()).map(|(a, b)| a * b).sum::<f32>() + raw.l2_b[r_idx];
        h2[k] = s.clamp(0.0, 1.0);
    }
    let orow = &raw.out_w[bk * out_in..(bk + 1) * out_in];
    let mlp = if v2 {
        orow[..l1].iter().zip(&h1).map(|(a, b)| a * b).sum::<f32>()
            + orow[l1..].iter().zip(&h2).map(|(a, b)| a * b).sum::<f32>()
            + raw.out_b[bk]
    } else {
        orow.iter().zip(&h2).map(|(a, b)| a * b).sum::<f32>() + raw.out_b[bk]
    };
    (bk, h1, h2, mlp + 0.5 * (ps[0] - ps[1]))
}

/// D1267: Revive dead CReLU L2 slots (and empty-bucket twins) with orthogonalized active directions
/// while setting outgoing out_w/dis_w to 0.0 so step-0 network output is 100% bit-identical (`C12/C14`).
fn resurrect_net(net_path: &str, val_path: &str, n_val: usize, out_path: &str) {
    let mut raw = RawTnn::load(net_path).expect("load raw tnn");
    let nb = raw.buckets;
    let w = raw.width;
    let l1 = raw.l1;
    let l2 = raw.l2;
    let v2 = &raw.magic == b"TNN2";
    let din = if raw.pairwise { w } else { 2 * w };
    let out_in = if v2 { l1 + l2 } else { l2 };

    println!("=== WEIGHT-SPACE STEP-0 DEAD-CRELU RESURRECTION: {net_path} -> {out_path} ===");
    let rows = read_strided_train(val_path, n_val);
    let mut bk_h1: Vec<Vec<Vec<f32>>> = vec![Vec::new(); nb];
    let mut h1_sum = vec![vec![0.0f64; l1]; nb];
    let mut h1_sum2 = vec![vec![0.0f64; l1]; nb];
    let mut h2_nz = vec![vec![0u64; l2]; nb];
    let mut bk_n = vec![0u64; nb];

    let mut acc = vec![0f32; 2 * w];
    let mut x = vec![0f32; din];
    for r in &rows {
        let b = board_of(r);
        let (bk, h1, h2, _) = eval_h1_h2(&raw, &b, &mut acc, &mut x);
        bk_n[bk] += 1;
        for j in 0..l1 {
            h1_sum[bk][j] += h1[j] as f64;
            h1_sum2[bk][j] += (h1[j] as f64) * (h1[j] as f64);
        }
        for k in 0..l2 {
            if h2[k] > 0.0 {
                h2_nz[bk][k] += 1;
            }
        }
        bk_h1[bk].push(h1);
    }

    let mut dead_by_bk: Vec<Vec<usize>> = Vec::with_capacity(nb);
    for bk in 0..nb {
        if bk_n[bk] == 0 {
            dead_by_bk.push(Vec::new());
            continue;
        }
        let d: Vec<usize> = (0..l2).filter(|&k| h2_nz[bk][k] == 0).collect();
        dead_by_bk.push(d);
    }

    let mut revived = 0usize;
    let mut prev_u_by_k: Vec<Vec<f64>> = vec![vec![0.0f64; l1]; l2];
    for bk in 0..nb {
        let ref_bk = if bk_n[bk] > 0 {
            bk
        } else if bk > 0 && bk_n[bk - 1] > 0 {
            bk - 1
        } else {
            continue;
        };
        let dead_list = if bk_n[bk] > 0 {
            dead_by_bk[bk].clone()
        } else {
            dead_by_bk[ref_bk].clone()
        };
        let n = bk_n[ref_bk] as f64;
        let active_h1: Vec<bool> = (0..l1)
            .map(|j| {
                let m = h1_sum[ref_bk][j] / n;
                (h1_sum2[ref_bk][j] / n - m * m).max(0.0).sqrt() > 1e-3
            })
            .collect();

        let mut chosen_in_bk: Vec<Vec<f64>> = Vec::new();
        for &k in &dead_list {
            let mut u = vec![0.0f64; l1];
            for j in 0..l1 {
                if active_h1[j] {
                    let ang = ((bk * 37 + k * 13 + j * 7 + 1) as f64) * 1.61803398875;
                    u[j] = ang.sin();
                }
            }
            // Gram-Schmidt orthogonalize against previously revived vectors in same bucket & twin bucket
            if bk % 2 == 1 && prev_u_by_k[k].iter().any(|&v| v != 0.0) {
                let dot: f64 = u.iter().zip(&prev_u_by_k[k]).map(|(a, b)| a * b).sum();
                for j in 0..l1 {
                    u[j] -= dot * prev_u_by_k[k][j];
                }
            }
            for prev in &chosen_in_bk {
                let dot: f64 = u.iter().zip(prev).map(|(a, b)| a * b).sum();
                for j in 0..l1 {
                    u[j] -= dot * prev[j];
                }
            }
            let norm = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm < 1e-6 {
                u.fill(0.0);
                if let Some(j0) = active_h1.iter().position(|&a| a) {
                    u[j0] = 1.0;
                } else {
                    u[0] = 1.0;
                }
            } else {
                for v in &mut u {
                    *v /= norm;
                }
            }
            prev_u_by_k[k] = u.clone();
            chosen_in_bk.push(u.clone());

            let mut m = 0.0f64;
            let mut m2 = 0.0f64;
            for h1 in &bk_h1[ref_bk] {
                let d: f64 = u.iter().zip(h1).map(|(&a, &b)| a * (b as f64)).sum();
                m += d;
                m2 += d * d;
            }
            m /= bk_h1[ref_bk].len() as f64;
            let sd = (m2 / bk_h1[ref_bk].len() as f64 - m * m).max(1e-6).sqrt();
            let max_u = u.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-6);
            let target_mean = 0.35f64;
            let scale = (0.30 / sd).min(1.85 / max_u).min(1.40 / m.abs().max(1e-6));

            let r_idx = bk * l2 + k;
            for j in 0..l1 {
                raw.l2_w[r_idx * l1 + j] = (u[j] * scale) as f32;
            }
            raw.l2_b[r_idx] = (target_mean - m * scale) as f32;
            raw.out_w[bk * out_in + if v2 { l1 + k } else { k }] = 0.0;
            if !raw.dis_w.is_empty() {
                raw.dis_w[bk * (l1 + l2) + l1 + k] = 0.0;
            }
            revived += 1;
        }
    }

    println!("  revived dead L2 slots (with out_w=0.0, dis_w=0.0): {revived}");
    raw.save(out_path).expect("save resurrected net");

    let base = Nnue::load(net_path).expect("load base net");
    let res = Nnue::load(out_path).expect("load resurrected net");
    let mut max_y_diff = 0.0f32;
    let mut max_dis_diff = 0i32;
    for r in rows.iter().take(2000) {
        let b = board_of(r);
        let (y0, d0) = base.forward_with_dis(&b, 0.0);
        let (y1, d1) = res.forward_with_dis(&b, 0.0);
        max_y_diff = max_y_diff.max((y0 - y1).abs());
        max_dis_diff = max_dis_diff.max((d0 - d1).abs());
    }
    println!(
        "  step-0 parity on 2000 boards: max |y0-y1| = {max_y_diff:.9} | max |dis0-dis1| = {max_dis_diff}"
    );
    audit_net(out_path, Some(val_path), n_val);
}

/// D1227/D1267 (`R1`): Per-bucket through-zero cp scale match of NET against REF (`nnue_main.tnn`) + TNN3 `dis` refit.
fn scale_match_net(net_path: &str, ref_path: &str, train_path: &str, val_path: &str, out_path: &str) {
    let mut raw = RawTnn::load(net_path).expect("load raw net");
    let net_a = Nnue::load(net_path).expect("load net_a");
    let net_ref = Nnue::load(ref_path).expect("load net_ref");
    let nb = raw.buckets;
    let l1 = raw.l1;
    let l2 = raw.l2;
    let v2 = &raw.magic == b"TNN2";
    let out_in = if v2 { l1 + l2 } else { l2 };

    println!("=== R1 SCALE-MATCH TO REF: {net_path} -> ref {ref_path} ===");
    let val = read_strided_train(val_path, 100_000);
    let mut sum_ab = vec![0.0f64; nb];
    let mut sum_aa = vec![0.0f64; nb];
    let mut cnt = vec![0usize; nb];
    for r in &val {
        let b = board_of(r);
        let bk = raw.bucket(b.empty().count_ones() as usize);
        let ya = net_a.forward(&b, 0.0) as f64;
        if (ya * 400.0).abs() > 1500.0 {
            continue;
        }
        let yref = net_ref.forward(&b, 0.0) as f64;
        sum_ab[bk] += ya * yref;
        sum_aa[bk] += ya * ya;
        cnt[bk] += 1;
    }
    let mut scales = vec![1.0f64; nb];
    let mut prev_k = 1.0f64;
    for bk in 0..nb {
        let k = if cnt[bk] < 50 {
            if bk == 15 { prev_k } else { 1.0 }
        } else {
            (sum_ab[bk] / sum_aa[bk].max(1e-12)).clamp(0.70, 1.30)
        };
        scales[bk] = k;
        prev_k = k;
        let kf = k as f32;
        for w_v in &mut raw.out_w[bk * out_in..(bk + 1) * out_in] {
            *w_v *= kf;
        }
        raw.out_b[bk] *= kf;
        for f in 0..raw.num_feats {
            raw.psqt[f * nb + bk] *= kf;
        }
        if !raw.s57w.is_empty() {
            raw.s57w[bk] *= kf;
        }
        println!("  R1 bk {:>2} (n={:>5}): k = {:.4}", bk, cnt[bk], k);
    }

    if &raw.magic == b"TNN3" {
        let dim = l1 + l2;
        let pdim = dim + 1;
        let mut a = vec![vec![vec![0.0f64; pdim]; pdim]; nb];
        let mut rhs = vec![vec![0.0f64; pdim]; nb];
        let mut d_cnt = vec![0usize; nb];
        let mut sy2 = vec![0.0f64; nb];
        let train = read_strided_train(train_path, 200_000);
        let mut acc = vec![0f32; 2 * raw.width];
        let din = if raw.pairwise { raw.width } else { 2 * raw.width };
        let mut x = vec![0f32; din];
        let mut used = 0usize;
        for r in &train {
            if r.cp.abs() >= 96000.0 {
                continue;
            }
            let b = board_of(r);
            let (bk, h1, h2, y_scaled) = eval_h1_h2(&raw, &b, &mut acc, &mut x);
            let delta = ((400.0 * y_scaled - r.cp).abs().min(2000.0) / 400.0) as f64;
            let mut feat = Vec::with_capacity(pdim);
            feat.extend(h1.iter().map(|&v| v as f64));
            feat.extend(h2.iter().map(|&v| v as f64));
            feat.push(1.0);
            for p in 0..pdim {
                if feat[p] != 0.0 {
                    for q in p..pdim {
                        a[bk][p][q] += feat[p] * feat[q];
                    }
                    rhs[bk][p] += feat[p] * delta;
                }
            }
            d_cnt[bk] += 1;
            sy2[bk] += delta * delta;
            used += 1;
        }
        raw.dis_w = vec![0.0f32; nb * dim];
        raw.dis_b = vec![0.0f32; nb];
        println!("  refitting TNN3 dis head on {used} unproven rows:");
        for bk in 0..nb {
            if d_cnt[bk] < 50 {
                raw.dis_b[bk] = 0.5;
                println!("    dis bk {:>2}: n={}, baseline 0.5", bk, d_cnt[bk]);
                continue;
            }
            for p in 0..pdim {
                for q in 0..p {
                    a[bk][p][q] = a[bk][q][p];
                }
                a[bk][p][p] += 100.0;
            }
            let sol = solve_spd(a[bk].clone(), rhs[bk].clone());
            for k in 0..dim {
                raw.dis_w[bk * dim + k] = sol[k] as f32;
            }
            raw.dis_b[bk] = sol[dim] as f32;
            let mut waw = 0.0f64;
            for p in 0..pdim {
                for q in 0..pdim {
                    let orig_a = if p == q { a[bk][p][q] - 100.0 } else { a[bk][p][q] };
                    waw += sol[p] * orig_a * sol[q];
                }
            }
            let wv: f64 = (0..pdim).map(|p| sol[p] * rhs[bk][p]).sum();
            let rmse = ((waw - 2.0 * wv + sy2[bk]) / d_cnt[bk] as f64).max(0.0).sqrt() * 400.0;
            println!("    dis bk {:>2}: n={:>5}, bias={:+.4}, RMSE={:.1} cp", bk, d_cnt[bk], raw.dis_b[bk], rmse);
        }
    }

    raw.save(out_path).expect("save r1 scale-matched net");
    let reloaded = Nnue::load(out_path).expect("verify load");
    let (mut dis_sum, mut dis_max) = (0.0f64, 0);
    for r in val.iter().take(1000) {
        let (_, d) = reloaded.forward_with_dis(&board_of(r), 0.0);
        dis_sum += d as f64;
        dis_max = dis_max.max(d);
    }
    println!("  verified Nnue::load({out_path}) | 1k sample mean dis {:.1} cp (max {dis_max} cp)", dis_sum / 1000.0);
}

pub fn run(args: &[String]) -> ExitCode {
    if let Some(p) = arg(args, "--resurrect") {
        let vp = arg(args, "--val").expect("--resurrect requires --val VAL.tdat");
        let n_val: usize = arg(args, "--n-val").map_or(50_000, |v| v.parse().unwrap());
        let out = arg(args, "--out").expect("--resurrect requires --out OUT.tnn");
        resurrect_net(&p, &vp, n_val, &out);
        return ExitCode::SUCCESS;
    }
    if let Some(p) = arg(args, "--scale-match") {
        let ref_p = arg(args, "--ref").expect("--scale-match requires --ref REF.tnn");
        let tr = arg(args, "--train").expect("--scale-match requires --train TRAIN.tdat");
        let vp = arg(args, "--val").expect("--scale-match requires --val VAL.tdat");
        let out = arg(args, "--out").expect("--scale-match requires --out OUT.tnn");
        scale_match_net(&p, &ref_p, &tr, &vp, &out);
        return ExitCode::SUCCESS;
    }
    if let Some(p) = arg(args, "--escape") {
        let train_path = arg(args, "--train").expect("--train TRAIN.tdat");
        let val_path = arg(args, "--val").expect("--val VAL.tdat");
        let n_train: usize = arg(args, "--n-train").map_or(500_000, |v| v.parse().unwrap());
        let out_path = arg(args, "--out").expect("--out OUT.tnn");
        escape_fit(&p, &train_path, &val_path, n_train, &out_path);
        return ExitCode::SUCCESS;
    }
    if let Some(p) = arg(args, "--deep") {
        let vp = arg(args, "--val").expect("--deep requires --val VAL.tdat");
        let n_val: usize = arg(args, "--n-val").map_or(50_000, |v| v.parse().unwrap());
        let d4_out = arg(args, "--d4-out");
        deep_net_report(&p, &vp, n_val, d4_out.as_deref());
        return ExitCode::SUCCESS;
    }
    if let Some(p) = arg(args, "--audit") {
        let vp = arg(args, "--val");
        let n_val: usize = arg(args, "--n-val").map_or(50_000, |v| v.parse().unwrap());
        audit_net(&p, vp.as_deref(), n_val);
        if let Some(p2) = arg(args, "--compare") {
            println!();
            audit_net(&p2, vp.as_deref(), n_val);
        }
        return ExitCode::SUCCESS;
    }
    if let Some(p) = arg(args, "--inspect") {
        inspect_net(&p);
        if let Some(p2) = arg(args, "--compare") {
            inspect_net(&p2);
        }
        return ExitCode::SUCCESS;
    }
    let net_path = arg(args, "--fit").expect("--fit NET.tnn, --inspect NET.tnn, or --audit NET.tnn");
    let train_path = arg(args, "--train").expect("--train TRAIN.tdat");
    let val_path = arg(args, "--val").expect("--val VAL.tdat");
    let n_train: usize = arg(args, "--n-train").map_or(400_000, |v| v.parse().unwrap());
    let out_prefix = arg(args, "--out-prefix").expect("--out-prefix STEM");

    inspect_net(&net_path);
    let raw_orig = RawTnn::load(&net_path).expect("load raw");
    let base_net = Nnue::load(&net_path).expect("load nnue");
    println!("loading {n_train} strided rows from {train_path} and full val from {val_path}...");
    let train = read_strided_train(&train_path, n_train);
    let val = crate::hcetrain::read_rows(&val_path, 2_000_000);
    println!("loaded train={} val={}", train.len(), val.len());

    eval_val_report("BASE", &base_net, &val);

    // Precompute base y and features per train row
    struct Sample {
        bk: usize,
        y0: f32,
        t: f32,
        wt: f32,
        c_diff: f32,
        e_diff: f32,
        i_diff: f32,
    }
    let samples: Vec<Sample> = train
        .iter()
        .map(|r| {
            let b = board_of(r);
            let empty = b.empty();
            let sm = safe_mask(b.me(), empty);
            let so = safe_mask(b.opp(), empty);
            let bk = raw_orig.bucket(empty.count_ones() as usize);
            let y0 = base_net.forward(&b, 0.0);
            Sample {
                bk,
                y0,
                t: target(r, true),
                wt: r.wsum.max(0.0),
                c_diff: (sm & CORNERS).count_ones() as f32 - (so & CORNERS).count_ones() as f32,
                e_diff: (sm & !CORNERS & !INTERIOR_5X5).count_ones() as f32
                    - (so & !CORNERS & !INTERIOR_5X5).count_ones() as f32,
                i_diff: (sm & INTERIOR_5X5).count_ones() as f32 - (so & INTERIOR_5X5).count_ones() as f32,
            }
        })
        .collect();

    // 1. Fit Variant A: safe-stone ONLY deltas per bucket [dc, de, di] (K = 3), IRLS optimizing MSE / pow2.5
    let nb = raw_orig.buckets;
    let mut safe_deltas = vec![[0.0f64; 3]; nb];
    for bk in 0..nb {
        let mut p = [0.0f64; 3];
        for _iter in 0..6 {
            let mut a = [[0.0f64; 6]; 6];
            let mut rhs = [0.0f64; 6];
            for i in 0..3 {
                a[i][i] = 200.0; // Ridge regularization so small buckets stay safe
            }
            for s in samples.iter().filter(|s| s.bk == bk) {
                let x = [s.c_diff as f64, s.e_diff as f64, s.i_diff as f64];
                let y = s.y0 as f64 + p[0] * x[0] + p[1] * x[1] + p[2] * x[2];
                let sigy = 1.0 / (1.0 + (-y).exp());
                let g = sigy * (1.0 - sigy);
                let err = s.t as f64 - sigy;
                let w = s.wt as f64 * g * g;
                for r in 0..3 {
                    rhs[r] += s.wt as f64 * g * err * x[r];
                    for c in 0..3 {
                        a[r][c] += w * x[r] * x[c];
                    }
                }
            }
            let dp = solve_sym(a, rhs, 3);
            for i in 0..3 {
                p[i] += dp[i];
            }
        }
        safe_deltas[bk] = p;
        println!(
            "  fit safe-only bk {:>2}: corner {:+5.1} cp | edge {:+5.1} cp | int {:+5.1} cp",
            bk,
            p[0] * 400.0,
            p[1] * 400.0,
            p[2] * 400.0
        );
    }

    let mut raw_safe = raw_orig.clone();
    apply_safe_deltas(&mut raw_safe, &safe_deltas);
    let safe_path = format!("{out_prefix}_safe.tnn");
    raw_safe.save(&safe_path).expect("save safe");
    let net_safe = Nnue::load(&safe_path).expect("reload safe");
    eval_val_report("WS_SAFE", &net_safe, &val);
    paired_check("BASE", &base_net, "WS_SAFE", &net_safe, &val);

    // 2. Fit Variant B: per-bucket affine scale (1 + ds) & bias db PLUS safe deltas [ds, db, dc, de, di] (K = 5)
    let mut cal_params = vec![[0.0f64; 5]; nb];
    for bk in 0..nb {
        let mut p = [0.0f64; 5]; // [ds, db, dc, de, di] where y = (1 + ds)*y0 + db + ...
        for _iter in 0..6 {
            let mut a = [[0.0f64; 6]; 6];
            let mut rhs = [0.0f64; 6];
            a[0][0] = 500.0; // strong prior toward scale 1.0
            a[1][1] = 200.0;
            for i in 2..5 {
                a[i][i] = 200.0;
            }
            for s in samples.iter().filter(|s| s.bk == bk) {
                let x = [s.y0 as f64, 1.0, s.c_diff as f64, s.e_diff as f64, s.i_diff as f64];
                let y = s.y0 as f64 + p[0] * x[0] + p[1] + p[2] * x[2] + p[3] * x[3] + p[4] * x[4];
                let sigy = 1.0 / (1.0 + (-y).exp());
                let g = sigy * (1.0 - sigy);
                let err = s.t as f64 - sigy;
                let w = s.wt as f64 * g * g;
                for r in 0..5 {
                    rhs[r] += s.wt as f64 * g * err * x[r];
                    for c in 0..5 {
                        a[r][c] += w * x[r] * x[c];
                    }
                }
            }
            let dp = solve_sym(a, rhs, 5);
            for i in 0..5 {
                p[i] += dp[i];
            }
        }
        cal_params[bk] = p;
        println!(
            "  fit safe+cal  bk {:>2}: scale {:.3} bias {:+5.1} cp | corner {:+5.1} edge {:+5.1} int {:+5.1}",
            bk,
            1.0 + p[0],
            p[1] * 400.0,
            p[2] * 400.0,
            p[3] * 400.0,
            p[4] * 400.0
        );
    }

    let mut raw_cal = raw_orig.clone();
    let out_in = if &raw_cal.magic == b"TNN2" { raw_cal.l1 + raw_cal.l2 } else { raw_cal.l2 };
    for bk in 0..nb {
        let scale = (1.0 + cal_params[bk][0]) as f32;
        let bias = cal_params[bk][1] as f32;
        for w in &mut raw_cal.out_w[bk * out_in..(bk + 1) * out_in] {
            *w *= scale;
        }
        raw_cal.out_b[bk] = raw_cal.out_b[bk] * scale + bias;
        for f in 0..raw_cal.num_feats {
            raw_cal.psqt[f * nb + bk] *= scale;
        }
        if !raw_cal.s57w.is_empty() {
            raw_cal.s57w[bk] *= scale;
        }
    }
    let cal_safe: Vec<[f64; 3]> = cal_params.iter().map(|p| [p[2], p[3], p[4]]).collect();
    apply_safe_deltas(&mut raw_cal, &cal_safe);
    let cal_path = format!("{out_prefix}_safecal.tnn");
    raw_cal.save(&cal_path).expect("save safecal");
    let net_cal = Nnue::load(&cal_path).expect("reload safecal");
    eval_val_report("WS_SAFECAL", &net_cal, &val);
    paired_check("BASE", &base_net, "WS_SAFECAL", &net_cal, &val);
    inspect_net(&safe_path);

    ExitCode::SUCCESS
}
