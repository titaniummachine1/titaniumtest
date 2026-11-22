//! `lab jat3-train` (D1136/D1137, user: "do what Zataxx does but with 3x3, initialise from Zataxx's 2x2, let it learn
//! the rest"; Jataxx's input): a 3x3-window tuple NETWORK, not a linear sum.
//!
//!   inputs: the 25 overlapping 3x3 windows of the 7x7 board, each a base-3 code of its 9 cells (0 empty, 1 side to move,
//!           2 opponent), one input row per (window, code), untied: 25 x 19,683 = 492,075 rows;
//!   net:    h = fb + sum of the 25 rows (256 wide), a = clamp(h, 0, 1)^2 (SCReLU), out = sum a * ow + ob = cp / 400.
//!
//! C14 / C12 init = EXACT copy of a trained Zataxx 2x2 net (36 2x2 windows, 2916 -> 256 SCReLU -> 1): each 3x3 window
//! holds four 2x2 windows; the row of a 3x3 code = sum of the Zataxx rows of its four 2x2 sub-codes, each 2x2 window split
//! equally over the 3x3 windows that contain it (1 / 2 / 4 at corner / edge / interior), so the sum over the 25 rows equals
//! the sum over the 36 Zataxx rows on every board: step 0 == the Zataxx net (checked and logged). Training then learns what
//! the 3x3 shapes add.
//!
//! Data, target, loss, validation = the lab STANDARD (lab/STANDARD_TRAIN.txt, D1130): whole dataset, champion target
//! (hcetrain::target), |sigmoid(out) - t|^2.5 weighted by wsum, held-out tail; online D4 augmentation always on.
//! Optimiser: Adam; first-layer rows are updated only when a batch touches them (moment decay caught up exactly by
//! b^delta, the momentum drift of untouched steps is not applied = lazy Adam), the 513 head weights densely.
//! Saturation (LAWS C15, D1133 trap 5): lr halves after each epoch without a val gain; after 4 such epochs the run restarts
//! from the best weights with a fresh lr (lr0 / 2, fresh moments); it stops only when TWO restarts in a row bring no gain.
//! Outputs: {out}.log, {out}_jat3.bin (best weights, f32), {out}_jat3.json (stamp + path; eval-gate plays it).
use crate::extval::{own_opp, Zat};
use crate::hcetrain::{read_rows, sig, target};
use crate::util::{arg, Rec};
use crate::zattrain::{d4maps, sym};
use std::io::{Read, Write};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

/// Zataxx's hidden width (the graft source); a jat3 net's own width is in its file header (D1152: 256 or downscaled)
pub const ZH: usize = 256;
pub const MAXH: usize = 256;
pub const NW: usize = 25;
pub const NC: usize = 19_683;
pub const NI: usize = NW * NC;

/// weight layout of a jat3 net with hidden width h: [NI rows x h][fb h][ow h][ob]
#[derive(Clone, Copy)]
pub struct Lay {
    pub h: usize,
}

impl Lay {
    pub fn fb(&self) -> usize {
        NI * self.h
    }
    pub fn ow(&self) -> usize {
        NI * self.h + self.h
    }
    pub fn ob(&self) -> usize {
        NI * self.h + 2 * self.h
    }
    pub fn p(&self) -> usize {
        NI * self.h + 2 * self.h + 1
    }
}

/// base-3 value of one 3-cell window row: TAB[me 3 bits][opp 3 bits]
pub(crate) fn row_tab() -> [[u16; 8]; 8] {
    let mut t = [[0u16; 8]; 8];
    for m in 0..8usize {
        for o in 0..8usize {
            if m & o != 0 {
                continue;
            }
            let mut v = 0u16;
            for k in 0..3 {
                let d = if m >> k & 1 == 1 { 1 } else if o >> k & 1 == 1 { 2 } else { 0 };
                v += d * 3u16.pow(k as u32);
            }
            t[m][o] = v;
        }
    }
    t
}

/// the 25 input rows of a position (side to move = us); digit of cell (dr, dc) has weight 3^(3 dr + dc)
pub fn jidx(us: u64, them: u64, t: &[[u16; 8]; 8], out: &mut [u32; NW]) {
    for r in 0..5 {
        for c in 0..5 {
            let off = 7 * r + c;
            let mut code = 0u32;
            let mut mul = 1u32;
            for dr in 0..3 {
                let s = off + 7 * dr;
                code += mul * t[((us >> s) & 7) as usize][((them >> s) & 7) as usize] as u32;
                mul *= 27;
            }
            let w = 5 * r + c;
            out[w] = (w * NC) as u32 + code;
        }
    }
}

fn fwd(w: &[f32], ix: &[u32; NW], h: &mut [f32; MAXH], ly: Lay) -> f32 {
    let hd = ly.h;
    let (fb, ow, ob) = (ly.fb(), ly.ow(), ly.ob());
    h[..hd].copy_from_slice(&w[fb..fb + hd]);
    for &i in ix {
        let row = &w[i as usize * hd..i as usize * hd + hd];
        for k in 0..hd {
            h[k] += row[k];
        }
    }
    let owv = &w[ow..ow + hd];
    let mut s = w[ob];
    for k in 0..hd {
        let a = h[k].clamp(0.0, 1.0);
        s += a * a * owv[k];
    }
    s
}

/// float Zataxx forward (same arithmetic as zattrain), for the step-0 parity check
fn zat_fwd(z: &Zat, us: u64, them: u64) -> f32 {
    let mut ix = [0usize; 36];
    crate::zattrain::zidx_pub(us, them, &mut ix);
    let mut h = [0f32; ZH];
    for k in 0..ZH {
        h[k] = z.fb[k] as f32 / 255.0;
    }
    for &i in &ix {
        for k in 0..ZH {
            h[k] += z.fw[i * ZH + k] as f32 / 255.0;
        }
    }
    let mut s = z.obias as f32 / (255.0 * 64.0);
    for k in 0..ZH {
        let a = h[k].clamp(0.0, 1.0);
        s += a * a * (z.ow[k] as f32 / 64.0);
    }
    s
}

/// exact graft of a Zataxx 2x2 net into the 3x3 first layer (see the module doc)
fn graft(z: &Zat, w: &mut [f32], threads: usize) {
    const H: usize = ZH;
    let zl = Lay { h: ZH };
    let (fbo, owo, obo) = (zl.fb(), zl.ow(), zl.ob());
    let cnt = |i: usize| if i == 0 || i == 5 { 1.0f32 } else { 2.0 };
    let wins: Vec<(usize, &mut [f32])> = w[..fbo].chunks_mut(NC * H).enumerate().collect();
    let per = wins.len().div_ceil(threads.max(1));
    let mut groups: Vec<Vec<(usize, &mut [f32])>> = Vec::new();
    let mut it = wins.into_iter();
    loop {
        let g: Vec<_> = it.by_ref().take(per).collect();
        if g.is_empty() {
            break;
        }
        groups.push(g);
    }
    std::thread::scope(|sc| {
        for g in groups {
            sc.spawn(move || {
                for (wi, sl) in g {
                    let (r, c) = (wi / 5, wi % 5);
                    for code in 0..NC {
                        let mut d = [0usize; 9];
                        let mut x = code;
                        for v in d.iter_mut() {
                            *v = x % 3;
                            x /= 3;
                        }
                        let row = &mut sl[code * H..code * H + H];
                        row.iter_mut().for_each(|v| *v = 0.0);
                        for a in 0..2 {
                            for b in 0..2 {
                                let (i, j) = (r + a, c + b);
                                let sub = d[3 * a + b] + 3 * d[3 * a + b + 1] + 9 * d[3 * (a + 1) + b] + 27 * d[3 * (a + 1) + b + 1];
                                let zi = (6 * i + j) * 81 + sub;
                                let inv = 1.0 / (255.0 * cnt(i) * cnt(j));
                                for k in 0..H {
                                    row[k] += z.fw[zi * H + k] as f32 * inv;
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    for k in 0..H {
        w[fbo + k] = z.fb[k] as f32 / 255.0;
        w[owo + k] = z.ow[k] as f32 / 64.0;
    }
    w[obo] = z.obias as f32 / (255.0 * 64.0);
}

/// D1152 downscale = the hub's standard `ti train shrink --refit` method (shrink.rs) carried over to jat3: activations
/// feeding the next layer x_c = clamp(h_c, 0, 1)^2 (SCReLU), next layer = the single output unit; channel choice =
/// pivoted Cholesky on the downstream-scaled covariance (cov of x times |ow_p| |ow_q|); ridge refit (1e-4 x mean diagonal)
/// of ow + ob on the kept channels' x, target = the original net's output; fit on the even calibration rows, report on
/// the odd ones. Returns (new weights, report line).
fn shrink_refit(net: &Jat3Net, h2: usize, cal: &[(u64, u64)]) -> (Vec<f32>, String) {
    let (ly, hd) = (net.ly, net.ly.h);
    assert!(h2 > 0 && h2 < hd, "--shrink must be below the current width {hd}");
    let mut buf = [0f32; MAXH];
    let mut ix = [0u32; NW];
    // activations + original output per calibration position
    let mut xs: Vec<f32> = Vec::with_capacity(cal.len() * hd);
    let mut ys: Vec<f64> = Vec::with_capacity(cal.len());
    for &(me, op) in cal {
        jidx(me, op, &net.t, &mut ix);
        ys.push(fwd(&net.w, &ix, &mut buf, ly) as f64);
        xs.extend(buf[..hd].iter().map(|v| {
            let a = v.clamp(0.0, 1.0);
            a * a
        }));
    }
    let owv: Vec<f64> = net.w[ly.ow()..ly.ow() + hd].iter().map(|&v| v as f64).collect();
    let fit: Vec<usize> = (0..cal.len()).step_by(2).collect();
    let test: Vec<usize> = (1..cal.len()).step_by(2).collect();
    // downstream-scaled covariance on the fit rows
    let mut mean = vec![0f64; hd];
    for &r in &fit {
        for c in 0..hd {
            mean[c] += xs[r * hd + c] as f64;
        }
    }
    mean.iter_mut().for_each(|m| *m /= fit.len() as f64);
    let mut cov = vec![vec![0f64; hd]; hd];
    let mut d = vec![0f64; hd];
    for &r in &fit {
        for c in 0..hd {
            d[c] = xs[r * hd + c] as f64 - mean[c];
        }
        for p in 0..hd {
            if d[p] != 0.0 {
                for q in p..hd {
                    cov[p][q] += d[p] * d[q];
                }
            }
        }
    }
    for p in 0..hd {
        for q in p..hd {
            let v = cov[p][q] / fit.len() as f64 * owv[p].abs() * owv[q].abs();
            cov[p][q] = v;
            cov[q][p] = v;
        }
    }
    // pivoted Cholesky (same as shrink.rs::pivots)
    let mut diag: Vec<f64> = (0..hd).map(|i| cov[i][i]).collect();
    let total: f64 = diag.iter().sum();
    let mut lcols: Vec<Vec<f64>> = Vec::new();
    let mut used = vec![false; hd];
    let mut keep = Vec::new();
    for _ in 0..h2 {
        let (mut p, mut best) = (usize::MAX, -1.0);
        for i in 0..hd {
            if !used[i] && diag[i] > best {
                best = diag[i];
                p = i;
            }
        }
        if p == usize::MAX || best <= 1e-18 {
            break;
        }
        used[p] = true;
        keep.push(p);
        let mut col = vec![0f64; hd];
        for i in 0..hd {
            let mut s = cov[i][p];
            for lj in &lcols {
                s -= lj[i] * lj[p];
            }
            col[i] = s / best.sqrt();
        }
        for i in 0..hd {
            diag[i] = (diag[i] - col[i] * col[i]).max(0.0);
        }
        lcols.push(col);
    }
    assert_eq!(keep.len(), h2, "only {} usable channels", keep.len());
    let unexplained: f64 = (0..hd).filter(|i| !used[*i]).map(|i| diag[i]).sum::<f64>() / total.max(1e-18);
    keep.sort_unstable();
    // ridge refit of ow + ob (target = original output) on the fit rows
    let n = h2 + 1;
    let mut gm = vec![vec![0f64; n]; n];
    let mut hv = vec![0f64; n];
    let mut xr = vec![0f64; n];
    for &r in &fit {
        for (j, &c) in keep.iter().enumerate() {
            xr[j] = xs[r * hd + c] as f64;
        }
        xr[h2] = 1.0;
        for p in 0..n {
            if xr[p] != 0.0 {
                for q in p..n {
                    gm[p][q] += xr[p] * xr[q];
                }
                hv[p] += xr[p] * ys[r];
            }
        }
    }
    for p in 0..n {
        for q in 0..p {
            gm[p][q] = gm[q][p];
        }
    }
    let tr: f64 = (0..n).map(|i| gm[i][i]).sum::<f64>() / n as f64;
    let ridge = 1e-4 * tr.max(1e-9);
    for i in 0..n {
        gm[i][i] += ridge;
    }
    // Gaussian elimination with partial pivoting (shrink.rs::solve_multi, one right-hand side)
    let mut b: Vec<f64> = hv;
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| gm[x][c].abs().partial_cmp(&gm[y][c].abs()).unwrap()).unwrap();
        gm.swap(c, p);
        b.swap(c, p);
        let dv = gm[c][c];
        for k in c..n {
            gm[c][k] /= dv;
        }
        b[c] /= dv;
        for r in 0..n {
            if r != c {
                let f = gm[r][c];
                if f != 0.0 {
                    for k in c..n {
                        let t = gm[c][k];
                        gm[r][k] -= f * t;
                    }
                    let t = b[c];
                    b[r] -= f * t;
                }
            }
        }
    }
    // held-out (odd rows) error of: plain prune (kept channels, original ow/ob) and the refit, in cp
    let obv = net.w[ly.ob()] as f64;
    let (mut ep, mut er) = (0f64, 0f64);
    for &r in &test {
        let (mut sp, mut sr) = (obv, b[h2]);
        for (j, &c) in keep.iter().enumerate() {
            let x = xs[r * hd + c] as f64;
            sp += owv[c] * x;
            sr += b[j] * x;
        }
        ep += ((sp - ys[r]) * 400.0).powi(2);
        er += ((sr - ys[r]) * 400.0).powi(2);
    }
    let (ep, er) = ((ep / test.len() as f64).sqrt(), (er / test.len() as f64).sqrt());
    // new weights: kept columns of every first-layer row and of fb, refit ow / ob
    let nl = Lay { h: h2 };
    let mut w2 = vec![0f32; nl.p()];
    for row in 0..NI {
        let src = &net.w[row * hd..row * hd + hd];
        let dst = &mut w2[row * h2..row * h2 + h2];
        for (j, &c) in keep.iter().enumerate() {
            dst[j] = src[c];
        }
    }
    for (j, &c) in keep.iter().enumerate() {
        w2[nl.fb() + j] = net.w[ly.fb() + c];
        w2[nl.ow() + j] = b[j] as f32;
    }
    w2[nl.ob()] = b[h2] as f32;
    let rep = format!(
        "downscale {hd} -> {h2} (ti train shrink --refit method): {} calibration positions (fit {} even / test {} odd); downstream-weighted variance NOT explained by the kept channels {:.3}%; held-out output RMSE vs the {hd} net: plain prune {ep:.1} cp, ridge refit {er:.1} cp",
        cal.len(),
        fit.len(),
        test.len(),
        100.0 * unexplained
    );
    (w2, rep)
}

/// a trained jat3 net for eval-gate (`*_jat3.json` -> its `_jat3.bin`)
pub struct Jat3Net {
    pub(crate) w: Vec<f32>,
    t: [[u16; 8]; 8],
    pub(crate) ly: Lay,
}

impl Jat3Net {
    pub fn load(json: &str) -> Jat3Net {
        let bin = json.trim_end_matches(".json").to_string() + ".bin";
        let mut f = std::fs::File::open(&bin).unwrap_or_else(|e| panic!("{bin}: {e}"));
        let mut b = Vec::new();
        f.read_to_end(&mut b).expect("read jat3 bin");
        assert_eq!(&b[..4], b"JAT3", "{bin}: not a JAT3 file");
        let h = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
        assert!(h > 0 && h <= MAXH, "{bin}: hidden width {h}");
        let ly = Lay { h };
        let w: Vec<f32> = b[8..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(w.len(), ly.p(), "{bin}: {} weights, expected {} for width {h}", w.len(), ly.p());
        Jat3Net { w, t: row_tab(), ly }
    }
    pub fn eval(&self, b: &Board) -> i32 {
        self.eval_f(b).round().clamp(-30000.0, 30000.0) as i32
    }
    /// unrounded cp (D1196: paired scoring uses the same value training scored)
    pub fn eval_f(&self, b: &Board) -> f32 {
        let mut ix = [0u32; NW];
        jidx(b.me(), b.opp(), &self.t, &mut ix);
        let mut h = [0f32; MAXH];
        fwd(&self.w, &ix, &mut h, self.ly) * 400.0
    }
}

/// one size option: time per eval (index extraction + accumulate + SCReLU + output), f32 and i16 tables
fn bench_f32<const W: usize>(rows: usize, map: Option<&[u32]>, pos: &[(u64, u64)], tab: &[[u16; 8]; 8]) -> (f64, f64) {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 * 0.02 - 0.01
    };
    let t: Vec<f32> = (0..rows * W).map(|_| rnd()).collect();
    let fb: Vec<f32> = (0..W).map(|_| 0.5 + rnd()).collect();
    let ow: Vec<f32> = (0..W).map(|_| rnd()).collect();
    let mut ix = [0u32; NW];
    let mut sink = 0f32;
    let t0 = std::time::Instant::now();
    for &(me, op) in pos {
        jidx(me, op, tab, &mut ix);
        let mut h = [0f32; W];
        h.copy_from_slice(&fb);
        for &i in &ix {
            let r = map.map_or(i, |m| m[i as usize]) as usize;
            let row = &t[r * W..r * W + W];
            for k in 0..W {
                h[k] += row[k];
            }
        }
        let mut o = 0f32;
        for k in 0..W {
            let a = h[k].clamp(0.0, 1.0);
            o += a * a * ow[k];
        }
        sink += o;
    }
    let ns = t0.elapsed().as_nanos() as f64 / pos.len() as f64;
    std::hint::black_box(sink);
    (ns, (rows * W * 4) as f64 / 1e6)
}

fn bench_i16<const W: usize>(rows: usize, map: Option<&[u32]>, pos: &[(u64, u64)], tab: &[[u16; 8]; 8]) -> (f64, f64) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 50) as i32 - 8192) as i16 / 64
    };
    let t: Vec<i16> = (0..rows * W).map(|_| rnd()).collect();
    let fb: Vec<i16> = (0..W).map(|_| 128 + rnd()).collect();
    let ow: Vec<i16> = (0..W).map(|_| rnd()).collect();
    let mut ix = [0u32; NW];
    let mut sink = 0i64;
    let t0 = std::time::Instant::now();
    for &(me, op) in pos {
        jidx(me, op, tab, &mut ix);
        let mut h = [0i16; W];
        h.copy_from_slice(&fb);
        for &i in &ix {
            let r = map.map_or(i, |m| m[i as usize]) as usize;
            let row = &t[r * W..r * W + W];
            for k in 0..W {
                h[k] = h[k].wrapping_add(row[k]);
            }
        }
        let mut o = 0i32;
        for k in 0..W {
            let a = h[k].clamp(0, 255) as i32;
            o = o.wrapping_add(a * a * ow[k] as i32);
        }
        sink += o as i64;
    }
    let ns = t0.elapsed().as_nanos() as f64 / pos.len() as f64;
    std::hint::black_box(sink);
    (ns, (rows * W * 2) as f64 / 1e6)
}

/// `lab jat3-bench [--n 200000]` (D1142): ns per eval (single thread, lab code, full refresh) of each 3x3-network size
/// option on real positions, next to the champion's eval on the same positions. Weights are random: only the arithmetic
/// and the memory traffic are timed, not the quality.
pub fn bench(args: &[String]) -> ExitCode {
    let n = arg(args, "--n").and_then(|v| v.parse().ok()).unwrap_or(200_000usize);
    let val = read_rows("C:/gitProjects/titanium-ataxx/data/nnue/packed/val_v9_clean.tdat", n);
    let pos: Vec<(u64, u64)> = val.iter().map(own_opp).collect();
    let tab = row_tab();
    // tied map: my index (window * NC + code, digit of cell (dr, dc) = 3^(3 dr + dc)) -> D4 class of win.rs
    let wn = crate::win::Win::new("3");
    let ct = wn.class_table();
    let mut map = vec![0u32; NI];
    for wi in 0..NW {
        for code in 0..NC {
            let mut x = code;
            let mut wc = 0usize;
            let mut d = [0usize; 9];
            for v in d.iter_mut() {
                *v = x % 3;
                x /= 3;
            }
            for &di in &d {
                wc = wc * 3 + di;
            }
            map[wi * NC + code] = ct[wi * NC + wc];
        }
    }
    let ntied = wn.dims();
    let net = Nnue::bundled();
    let boards: Vec<Board> = pos.iter().map(|&(a, b)| Board::from_occ(a, b, 0, 0)).collect();
    let mut sink = 0i64;
    let t0 = std::time::Instant::now();
    for b in &boards {
        sink += net.eval(b, 0) as i64;
    }
    let champ = t0.elapsed().as_nanos() as f64 / boards.len() as f64;
    std::hint::black_box(sink);
    println!("jat3-bench: {} positions (val_v9_clean), single thread, full refresh, lab code; tied classes {ntied}", pos.len());
    println!("champion NNUE eval (engine code, full refresh): {champ:.0} ns/eval");
    println!("{:<38} {:>10} {:>12} {:>12}", "option", "weights", "f32 ns|MB", "i16 ns|MB");
    let rows: [(&str, usize, Option<&[u32]>, usize); 4] = [
        ("untied 25x19683 rows, 256 hidden (ran)", NI, None, 256),
        ("1: untied, 128 hidden (Jataxx size)", NI, None, 128),
        ("2: D4-tied rows, 256 hidden", ntied, Some(&map), 256),
        ("3: D4-tied rows, 128 hidden", ntied, Some(&map), 128),
    ];
    for (name, r, m, h) in rows {
        let (f, i) = if h == 256 { (bench_f32::<256>(r, m, &pos, &tab), bench_i16::<256>(r, m, &pos, &tab)) } else { (bench_f32::<128>(r, m, &pos, &tab), bench_i16::<128>(r, m, &pos, &tab)) };
        println!("{name:<38} {:>10} {:>6.0} {:>5.0} {:>6.0} {:>5.0}", r * h + 2 * h + 1, f.0, f.1, i.0, i.1);
    }
    ExitCode::SUCCESS
}

/// `lab jat3-compare --model X_jat3.bin` (D1144): on the standard held-out rows, by empties band: weighted loss of the
/// 3x3 net vs the champion, the best logit blend of both (how much each knows that the other does not), the correlation
/// of their errors, and how often their sibling best moves differ.
pub fn compare(args: &[String]) -> ExitCode {
    let Some(mp) = arg(args, "--model") else {
        eprintln!("usage: lab jat3-compare --model X_jat3.bin");
        return ExitCode::from(2);
    };
    let (st, _tr, val) = match crate::standard::standard_split(args, &["--max-rows", "--val", "--data", "--holdout"]) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("jat3-compare: {e}");
            return ExitCode::from(2);
        }
    };
    drop(_tr);
    let j = Jat3Net::load(&mp.replace("_jat3.bin", "_jat3.json"));
    let net = Nnue::bundled();
    println!("{}", crate::standard::stamp(&None, &st, 0, val.len()));
    // per row: (band, weight, target, champion logit, jat3 logit)
    let bands = [(0u32, 6u32), (7, 12), (13, 24), (25, 36), (37, 49)];
    let rows: Vec<(usize, f32, f32, f32, f32)> = val
        .iter()
        .map(|r| {
            let (me, op) = own_opp(r);
            let b = Board::from_occ(me, op, 0, 0);
            let e = b.empty().count_ones();
            let bi = bands.iter().position(|&(lo, hi)| e >= lo && e <= hi).unwrap_or(4);
            (bi, r.wsum.max(0.0), target(r, false), net.eval(&b, 0) as f32 / 400.0, j.eval(&b) as f32 / 400.0)
        })
        .collect();
    let loss = |sel: &dyn Fn(usize) -> bool, a: f32| -> (f64, f64) {
        let (mut l, mut ws) = (0f64, 0f64);
        for &(bi, w, t, yc, yj) in &rows {
            if sel(bi) {
                let p = sig((1.0 - a) * yc + a * yj);
                l += w as f64 * ((p - t).abs() as f64).powf(2.5);
                ws += w as f64;
            }
        }
        (l / ws.max(1e-9), ws)
    };
    println!("{:<12} {:>9} {:>10} {:>10} {:>9} {:>16} {:>9}", "empties", "rows", "champion", "3x3 net", "3x3 vs C", "best blend (a)", "err corr");
    for bi in 0..=bands.len() {
        let sel = |x: usize| bi == bands.len() || x == bi;
        let name = if bi == bands.len() { "ALL".to_string() } else { format!("{}-{}", bands[bi].0, bands[bi].1) };
        let n = rows.iter().filter(|r| sel(r.0)).count();
        let (lc, _) = loss(&sel, 0.0);
        let (lj, _) = loss(&sel, 1.0);
        let mut bb = (lc, 0.0f32);
        for k in 1..=20 {
            let a = k as f32 * 0.05;
            let (l, _) = loss(&sel, a);
            if l < bb.0 {
                bb = (l, a);
            }
        }
        // correlation of the two errors (sigmoid - target), weighted
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy, mut sw) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for &(b2, w, t, yc, yj) in &rows {
            if sel(b2) {
                let (x, y, w) = ((sig(yc) - t) as f64, (sig(yj) - t) as f64, w as f64);
                sx += w * x;
                sy += w * y;
                sxx += w * x * x;
                syy += w * y * y;
                sxy += w * x * y;
                sw += w;
            }
        }
        let cov = sxy / sw - (sx / sw) * (sy / sw);
        let corr = cov / ((sxx / sw - (sx / sw).powi(2)).sqrt() * (syy / sw - (sy / sw).powi(2)).sqrt());
        println!("{name:<12} {n:>9} {lc:>10.6} {lj:>10.6} {:>8.2}% {:>10.6} ({:.2}) {corr:>9.3}", 100.0 * (lj / lc - 1.0), bb.0, bb.1);
    }
    // sibling best-move disagreement (5,000 held-out positions)
    let (mut dif, mut n) = (0f64, 0f64);
    let mut ml = MoveList::new();
    for r in val.iter().step_by((val.len() / 5000).max(1)).take(5000) {
        let (me, op) = own_opp(r);
        let b = Board::from_occ(me, op, 0, 0);
        b.legal_moves_dedup_into(&mut ml);
        if ml.len() < 2 {
            continue;
        }
        let (mut a, mut c) = ((i32::MIN, 0usize), (i32::MIN, 0usize));
        for (i, mv) in ml.iter().enumerate() {
            let ch = b.make(mv);
            let (yn, yj) = (-net.eval(&ch, 0), -j.eval(&ch));
            if yn > a.0 {
                a = (yn, i);
            }
            if yj > c.0 {
                c = (yj, i);
            }
        }
        dif += (a.1 != c.1) as u8 as f64;
        n += 1.0;
    }
    println!("sibling best move differs between the two nets in {:.2}% of {} held-out positions", 100.0 * dif / n, n);
    ExitCode::SUCCESS
}

fn save(out: &str, w: &[f32], h: usize) {
    let mut bytes = Vec::with_capacity(8 + w.len() * 4);
    bytes.extend_from_slice(b"JAT3");
    bytes.extend_from_slice(&(h as u32).to_le_bytes());
    for x in w {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    let tmp = format!("{out}_jat3.bin.tmp");
    let _ = std::fs::File::create(&tmp).and_then(|mut f| f.write_all(&bytes));
    let _ = std::fs::rename(&tmp, format!("{out}_jat3.bin"));
}

pub fn run(args: &[String]) -> ExitCode {
    let exp = crate::standard::experiment(args);
    let Some(zp) = arg(args, "--zataxx") else {
        eprintln!("usage: lab jat3-train --zataxx ZATAXX_NET.bin (init, C14) [--init PREV_jat3.bin] [--threads 7] [--bs 16384] [--lr 0.001] [--out PREFIX]");
        return ExitCode::from(2);
    };
    let g = |k: &str, d: f64| arg(args, k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let threads = (g("--threads", 7.0) as usize).max(1);
    let bs = g("--bs", 16384.0) as usize;
    let lr0 = g("--lr", 0.001) as f32;
    for banned in ["--max-epochs", "--epochs", "--steps"] {
        if arg(args, banned).is_some() || args.iter().any(|a| a == banned) {
            eprintln!("jat3-train: {banned} is refused (LAWS C15: training stops only by saturation)");
            return ExitCode::from(2);
        }
    }
    let out = arg(args, "--out").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/logs/hce/jat3".into());
    if let Err(e) = crate::standard::check_out(&out, &exp) {
        eprintln!("jat3-train: {e}");
        return ExitCode::from(2);
    }
    let t0 = std::time::Instant::now();
    // LAWS A14: a --resume keeps the earlier log and appends to it
    let mut log = if args.iter().any(|a| a == "--resume") { std::fs::read_to_string(format!("{out}.log")).unwrap_or_default() } else { String::new() };
    let say = |s: String, log: &mut String| {
        println!("{s}");
        log.push_str(&s);
        log.push('\n');
        let _ = std::fs::write(format!("{out}.log"), log.as_bytes());
    };
    let (train, val): (Vec<Rec>, Vec<Rec>) = if exp.is_none() {
        match crate::standard::standard_split(args, &["--max-rows", "--val", "--data", "--holdout"]) {
            Ok((st, tr, va)) => {
                say(crate::standard::stamp(&exp, &st, tr.len(), va.len()), &mut log);
                (tr, va)
            }
            Err(e) => {
                eprintln!("jat3-train: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        let (Some(data), Some(valp)) = (arg(args, "--data"), arg(args, "--val")) else {
            eprintln!("jat3-train --experiment: needs --data X --val V");
            return ExitCode::from(2);
        };
        let mr = g("--max-rows", 1e12) as usize;
        let (tr, va) = (read_rows(&data, mr), read_rows(&valp, 600_000));
        say(format!("EXPERIMENT {}: NOT the standard: {} train rows, {} val rows (data {data})", exp.clone().unwrap_or_default(), tr.len(), va.len()), &mut log);
        (tr, va)
    };
    let tab = row_tab();
    let maps = d4maps();
    let vix: Vec<[u32; NW]> = val.iter().map(|r| { let (me, op) = own_opp(r); let mut a = [0u32; NW]; jidx(me, op, &tab, &mut a); a }).collect();

    // init (C14): exact graft of the Zataxx 2x2 net, or a previous jat3 run (--init, resume)
    let z = Zat::load(&zp);
    // D1152: --shrink H2 (with --init) = downscale the previous net to H2 hidden units by the standard shrink --refit method
    let shrink_to = arg(args, "--shrink").and_then(|v| v.parse::<usize>().ok());
    let (mut w, ly, init_label) = if let Some(ip) = arg(args, "--init") {
        let prev = Jat3Net::load(&ip.replace("_jat3.bin", "_jat3.json"));
        if let Some(h2) = shrink_to {
            let cal: Vec<(u64, u64)> = train.iter().step_by((train.len() / 200_000).max(1)).take(200_000).map(own_opp).collect();
            let (w2, rep) = shrink_refit(&prev, h2, &cal);
            say(rep, &mut log);
            (w2, Lay { h: h2 }, format!("{ip} downscaled {} -> {h2} (standard shrink --refit method, C14)", prev.ly.h))
        } else {
            (prev.w.clone(), prev.ly, format!("previous jat3 run {ip}"))
        }
    } else {
        let zl = Lay { h: ZH };
        let mut w = vec![0f32; zl.p()];
        graft(&z, &mut w, threads);
        (w, zl, format!("exact graft of Zataxx 2x2 net {zp}"))
    };
    let hd = ly.h;
    let (fbo, owo) = (ly.fb(), ly.ow());
    let pn = ly.p();
    // step-0 parity with the Zataxx net (meaningful for a fresh graft only)
    let mut maxd = 0f32;
    let mut hh = [0f32; MAXH];
    for (r, ix) in val.iter().zip(&vix).step_by((val.len() / 20_000).max(1)) {
        let (me, op) = own_opp(r);
        let d = (fwd(&w, ix, &mut hh, ly) - zat_fwd(&z, me, op)).abs() * 400.0;
        maxd = maxd.max(d);
    }
    let vloss = |w: &[f32]| -> f64 {
        let chunk = val.len().div_ceil(threads);
        let parts: Vec<(f64, f64)> = std::thread::scope(|sc| {
            val.chunks(chunk)
                .zip(vix.chunks(chunk))
                .map(|(vs, xs)| {
                    sc.spawn(move || {
                        let mut h = [0f32; MAXH];
                        let (mut l, mut ws) = (0f64, 0f64);
                        for (r, ix) in vs.iter().zip(xs) {
                            let p = sig(fwd(w, ix, &mut h, ly));
                            let wt = r.wsum.max(0.0) as f64;
                            l += wt * ((p - target(r, false)).abs() as f64).powf(2.5);
                            ws += wt;
                        }
                        (l, ws)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        parts.iter().map(|p| p.0).sum::<f64>() / parts.iter().map(|p| p.1).sum::<f64>()
    };
    let net = Nnue::bundled();
    let champ = {
        let (mut l, mut ws) = (0f64, 0f64);
        for r in &val {
            let (me, op) = own_opp(r);
            let b = Board::from_occ(me, op, 0, 0);
            let p = sig(net.eval(&b, 0) as f32 / 400.0);
            let wt = r.wsum.max(0.0) as f64;
            l += wt * ((p - target(r, false)).abs() as f64).powf(2.5);
            ws += wt;
        }
        l / ws
    };
    let i_l = vloss(&w);
    say(format!("jat3-train: 3x3 tuple NETWORK 25 windows x {NC} codes = {NI} input rows -> {hd} SCReLU -> 1 ({pn} weights), {} train rows, {} val rows, {threads} threads, bs {bs}, lr {lr0}, D4 augmentation always on ({:.0}s)", train.len(), val.len(), t0.elapsed().as_secs_f64()), &mut log);
    say(format!("init = {init_label}: step-0 max |jat3 - zataxx| on 20k val positions {maxd:.4} cp (0 = exact graft); val loss {i_l:.6}; champion NNUE on the same rows {champ:.6}"), &mut log);

    let mut mo = vec![0f32; pn];
    let mut ve = vec![0f32; pn];
    let mut gfirst = vec![0f32; fbo];
    let mut last = vec![0u32; NI];
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut step = 0u32; // global batch counter (lazy-row bookkeeping)
    let mut astep = 0i32; // Adam bias-correction counter, reset with the moments at every restart
    let mut best = (i_l, w.clone(), 0usize);
    save(&out, &best.1, hd);
    let mut lr = lr0;
    let mut bad = 0usize;
    let mut cycle_gain = false;
    let mut zero_cycles = 0usize;
    let mut cycle = 0usize;
    let mut gh = vec![0f32; bs * hd];
    let mut ixb = vec![0u32; bs * NW];
    let mut small: Vec<Vec<f32>> = (0..threads).map(|_| vec![0f32; 2 * hd + 1]).collect();
    let nbatch = train.len().div_ceil(bs);
    // LAWS A14 (user D1307): best weights `{out}_jat3.bin` are already written from the start and on every new best; add a
    // full checkpoint `{out}.ckpt` after every epoch and on STOP (STOP in the out folder or C:/gitProjects/STOP_ALL,
    // checked before every batch: exact state saved, exit); `--resume` continues exactly. The epoch order is a pure
    // function of the epoch number.
    let ck_path = format!("{out}.ckpt");
    let run_dir = std::path::Path::new(&out).parent().map(|p| if p.as_os_str().is_empty() { std::path::PathBuf::from(".") } else { p.to_path_buf() }).unwrap_or_else(|| ".".into());
    let (mut ep_start, mut b_start, mut tl0, mut tw0) = (1usize, 0usize, 0f64, 0f64);
    let u32_as_f32 = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
    if args.iter().any(|a| a == "--resume") {
        let dd = match std::fs::read(&ck_path) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("jat3-train --resume: {ck_path}: {e}");
                return ExitCode::from(2);
            }
        };
        let mut r = crate::stopctl::Rd::new(&dd);
        let ok = (|| -> Result<(), String> {
            if r.str()? != "JCK1" || r.u64()? != pn as u64 || r.u64()? != hd as u64 {
                return Err("not a jat3-train checkpoint of this net".into());
            }
            ep_start = r.u64()? as usize;
            b_start = r.u64()? as usize;
            step = r.u64()? as u32;
            astep = r.u64()? as i32;
            lr = r.f64()? as f32;
            bad = r.u64()? as usize;
            cycle_gain = r.u64()? == 1;
            zero_cycles = r.u64()? as usize;
            cycle = r.u64()? as usize;
            tl0 = r.f64()?;
            tw0 = r.f64()?;
            best.0 = r.f64()?;
            best.2 = r.u64()? as usize;
            best.1 = r.f32s()?;
            w = r.f32s()?;
            mo = r.f32s()?;
            ve = r.f32s()?;
            last = r.f32s()?.into_iter().map(f32::to_bits).collect();
            Ok(())
        })();
        if let Err(e) = ok {
            eprintln!("jat3-train --resume: {ck_path}: {e}");
            return ExitCode::from(2);
        }
        say(format!("RESUMED from {ck_path}: epoch {ep_start} batch {b_start}, best epoch {} = {:.6}", best.2, best.0), &mut log);
    } else if std::path::Path::new(&ck_path).exists() {
        eprintln!("jat3-train: {ck_path} exists: continue it with --resume, or delete it to start over");
        return ExitCode::from(2);
    }
    #[allow(clippy::too_many_arguments)]
    fn save_ck(path: &str, pn: usize, hd: usize, ep: usize, bi: usize, step: u32, astep: i32, lr: f32, bad: usize, cg: bool, zc: usize, cy: usize, tl: f64, tw: f64, best: &(f64, Vec<f32>, usize), w: &[f32], mo: &[f32], ve: &[f32], last: &[f32]) -> Result<(), String> {
        let mut x = crate::stopctl::Wr::default();
        x.str("JCK1").u64(pn as u64).u64(hd as u64).u64(ep as u64).u64(bi as u64).u64(step as u64).u64(astep as u64).f64(lr as f64).u64(bad as u64).u64(cg as u64).u64(zc as u64).u64(cy as u64).f64(tl).f64(tw).f64(best.0).u64(best.2 as u64).f32s(&best.1).f32s(w).f32s(mo).f32s(ve).f32s(last);
        crate::stopctl::write_atomic(path, &x.0)
    }
    for epoch in ep_start.. {
        let (mut tl, mut tw) = if epoch == ep_start { (tl0, tw0) } else { (0f64, 0f64) };
        let mut order: Vec<usize> = (0..train.len()).collect();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ (epoch as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        for i in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        let te = std::time::Instant::now();
        let first = if epoch == ep_start { b_start } else { 0 };
        for (bi, batch) in order.chunks(bs).enumerate().skip(first) {
            if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
                if let Err(e) = save_ck(&ck_path, pn, hd, epoch, bi, step, astep, lr, bad, cycle_gain, zero_cycles, cycle, tl, tw, &best, &w, &mo, &ve, &u32_as_f32(&last)) {
                    eprintln!("jat3-train: checkpoint: {e}");
                    return ExitCode::from(1);
                }
                say(format!("STOPPED by {f} at epoch {epoch} batch {bi}: state {ck_path}, best (epoch {}) {out}_jat3.bin; the same command + --resume continues", best.2), &mut log);
                return ExitCode::SUCCESS;
            }
            let n = batch.len();
            let chunk = n.div_ceil(threads);
            // phase A: forward / backward per sample; per-sample first-layer gradient (gh) + indices, per-thread head grads
            let (wr, trr, mp, tb) = (&w, &train, &maps, &tab);
            let res: Vec<(f64, f64)> = std::thread::scope(|sc| {
                batch
                    .chunks(chunk)
                    .zip(gh.chunks_mut(chunk * hd))
                    .zip(ixb.chunks_mut(chunk * NW))
                    .zip(small.iter_mut())
                    .map(|(((idx, ghs), ixs), sg)| {
                        sc.spawn(move || {
                            let mut h = [0f32; MAXH];
                            let mut ix = [0u32; NW];
                            let (mut l, mut ws) = (0f64, 0f64);
                            for (s, &kk) in idx.iter().enumerate() {
                                let r = &trr[kk];
                                let wt = r.wsum.max(0.0);
                                let mut x = (kk as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (epoch as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                                x ^= x >> 29;
                                let m = &mp[(x.wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 61) as usize & 7];
                                let (me, op) = own_opp(r);
                                jidx(sym(me, m), sym(op, m), tb, &mut ix);
                                ixs[s * NW..s * NW + NW].copy_from_slice(&ix);
                                let g = &mut ghs[s * hd..s * hd + hd];
                                if wt == 0.0 {
                                    g.iter_mut().for_each(|v| *v = 0.0);
                                    continue;
                                }
                                let y = fwd(wr, &ix, &mut h, ly);
                                let p = sig(y);
                                let diff = p - target(r, true);
                                let ad = diff.abs();
                                l += (wt * ad.powf(2.5)) as f64;
                                ws += wt as f64;
                                let dy = wt * 2.5 * diff.signum() * ad.powf(1.5) * p * (1.0 - p);
                                sg[2 * hd] += dy;
                                for k in 0..hd {
                                    let hk = h[k];
                                    let a = hk.clamp(0.0, 1.0);
                                    sg[hd + k] += dy * a * a;
                                    let gk = if hk > 0.0 && hk < 1.0 { dy * wr[owo + k] * 2.0 * hk } else { 0.0 };
                                    g[k] = gk;
                                    sg[k] += gk;
                                }
                            }
                            (l, ws)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect()
            });
            let ws: f64 = res.iter().map(|p| p.1).sum::<f64>().max(1e-9);
            tl += res.iter().map(|p| p.0).sum::<f64>();
            tw += ws;
            step += 1;
            astep += 1;
            let (c1, c2) = (1.0 - b1.powi(astep), 1.0 - b2.powi(astep));
            let inv = 1.0 / ws as f32;
            // phase B: first layer, threads split by WINDOW (rows of different windows never collide): accumulate the
            // batch gradient of each touched row, then lazy Adam on exactly those rows
            {
                let mut per: Vec<Vec<(usize, &mut [f32], &mut [f32], &mut [f32], &mut [f32], &mut [u32])>> = (0..threads).map(|_| Vec::new()).collect();
                for (wi, ((((ws_, ms), vs), gs), ls)) in w[..fbo]
                    .chunks_mut(NC * hd)
                    .zip(mo[..fbo].chunks_mut(NC * hd))
                    .zip(ve[..fbo].chunks_mut(NC * hd))
                    .zip(gfirst.chunks_mut(NC * hd))
                    .zip(last.chunks_mut(NC))
                    .enumerate()
                {
                    per[wi % threads].push((wi, ws_, ms, vs, gs, ls));
                }
                let (ghr, ixr, lrv) = (&gh, &ixb, lr);
                std::thread::scope(|sc| {
                    for grp in per {
                        sc.spawn(move || {
                            let mut seen = vec![false; NC];
                            let mut touched: Vec<usize> = Vec::new();
                            for (wi, wsl, msl, vsl, gsl, lsl) in grp {
                                touched.clear();
                                for s in 0..n {
                                    let row = ixr[s * NW + wi] as usize - wi * NC;
                                    if !seen[row] {
                                        seen[row] = true;
                                        touched.push(row);
                                    }
                                    let gr = &mut gsl[row * hd..row * hd + hd];
                                    for (a, b) in gr.iter_mut().zip(&ghr[s * hd..s * hd + hd]) {
                                        *a += *b;
                                    }
                                }
                                for &row in &touched {
                                    seen[row] = false;
                                    let delta = (step - lsl[row]) as i32;
                                    lsl[row] = step;
                                    let (d1, d2) = (b1.powi(delta - 1) * b1, b2.powi(delta - 1) * b2);
                                    for k in row * hd..row * hd + hd {
                                        let gi = gsl[k] * inv;
                                        gsl[k] = 0.0;
                                        msl[k] = d1 * msl[k] + (1.0 - b1) * gi;
                                        vsl[k] = d2 * vsl[k] + (1.0 - b2) * gi * gi;
                                        wsl[k] -= lrv * (msl[k] / c1) / ((vsl[k] / c2).sqrt() + eps);
                                    }
                                }
                            }
                        });
                    }
                });
            }
            // phase C: head (fb, ow, ob), dense Adam
            for j in 0..2 * hd + 1 {
                let mut gi = 0f32;
                for sg in small.iter_mut() {
                    gi += sg[j];
                    sg[j] = 0.0;
                }
                gi *= inv;
                let p = fbo + j;
                mo[p] = b1 * mo[p] + (1.0 - b1) * gi;
                ve[p] = b2 * ve[p] + (1.0 - b2) * gi * gi;
                w[p] -= lr * (mo[p] / c1) / ((ve[p] / c2).sqrt() + eps);
            }
            if (bi + 1) % 500 == 0 {
                let done = (bi + 1) as f64 / nbatch as f64;
                say(format!("  epoch {epoch} batch {}/{nbatch}: train loss {:.6} | {:.0} rows/s | epoch ETA {:.0}s", bi + 1, tl / tw.max(1e-9), ((bi + 1) * bs) as f64 / te.elapsed().as_secs_f64(), te.elapsed().as_secs_f64() * (1.0 / done - 1.0)), &mut log);
            }
        }
        let vl = vloss(&w);
        let improved = vl < best.0 - 1e-7;
        if improved {
            best = (vl, w.clone(), epoch);
            save(&out, &best.1, hd);
            bad = 0;
            cycle_gain = true;
        } else {
            bad += 1;
            lr *= 0.5;
        }
        say(format!(
            "epoch {epoch}: train loss {:.6} | val loss {vl:.6} | best {:.6} @ {} | lr {lr:.6} | cycle {cycle} | champion {champ:.6} | {:.0}s",
            tl / tw.max(1e-9), best.0, best.2, t0.elapsed().as_secs_f64()
        ), &mut log);
        if bad >= 4 {
            if cycle_gain {
                zero_cycles = 0;
            } else {
                zero_cycles += 1;
            }
            if zero_cycles >= 2 {
                say(format!("SATURATED: two fresh-lr restarts in a row brought no val gain; best epoch {} val loss {:.6} (init {i_l:.6}, champion {champ:.6})", best.2, best.0), &mut log);
                break;
            }
            cycle += 1;
            w.copy_from_slice(&best.1);
            mo.iter_mut().for_each(|v| *v = 0.0);
            ve.iter_mut().for_each(|v| *v = 0.0);
            last.iter_mut().for_each(|v| *v = step);
            astep = 0;
            lr = lr0 * 0.5;
            bad = 0;
            cycle_gain = false;
            say(format!("RESTART {cycle}: from the best weights (epoch {}, {:.6}) with fresh lr {lr} and fresh Adam moments", best.2, best.0), &mut log);
        }
        if let Err(e) = save_ck(&ck_path, pn, hd, epoch + 1, 0, step, astep, lr, bad, cycle_gain, zero_cycles, cycle, 0.0, 0.0, &best, &w, &mo, &ve, &u32_as_f32(&last)) {
            eprintln!("jat3-train: checkpoint: {e}");
            return ExitCode::from(1);
        }
    }
    // agreement with the champion on sibling best moves (2,000 val positions), best weights
    let bn = Jat3Net { w: best.1.clone(), t: tab, ly };
    let (mut ag, mut nn) = (0f64, 0f64);
    let mut ml = MoveList::new();
    for r in val.iter().step_by((val.len() / 2000).max(1)).take(2000) {
        let (me, op) = own_opp(r);
        let b = Board::from_occ(me, op, 0, 0);
        b.legal_moves_dedup_into(&mut ml);
        if ml.len() < 2 {
            continue;
        }
        let (mut a, mut c) = ((i32::MIN, 0usize), (i32::MIN, 0usize));
        for (i, mv) in ml.iter().enumerate() {
            let ch = b.make(mv);
            let yn = -net.eval(&ch, 0);
            let yj = -bn.eval(&ch);
            if yn > a.0 {
                a = (yn, i);
            }
            if yj > c.0 {
                c = (yj, i);
            }
        }
        ag += (a.1 == c.1) as u8 as f64;
        nn += 1.0;
    }
    let stamp = match &exp {
        None => "\"standard\":true,\"experiment\":\"\"".to_string(),
        Some(n) => format!("\"standard\":false,\"experiment\":\"{n}\""),
    };
    let js = format!(
        "{{{stamp},\"model\":\"jat3\",\"hidden\":{hd},\"inputs\":{NI},\"train_rows\":{},\"val_rows\":{},\"best_epoch\":{},\"val_loss\":{},\"champion_val_loss\":{champ},\"init\":\"{}\"}}",
        train.len(),
        val.len(),
        best.2,
        best.0,
        init_label.replace('\\', "/")
    );
    let _ = std::fs::write(format!("{out}_jat3.json"), js);
    say(format!("best weights {out}_jat3.bin; sibling best-move agreement with the champion {:.2}% ({} positions)", 100.0 * ag / nn.max(1.0), nn), &mut log);
    ExitCode::SUCCESS
}
