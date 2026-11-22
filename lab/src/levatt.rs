//! `lab levatt-train [--no-attn] [--threads 7] [--bs 16384] [--lr 0.0005] --out PREFIX` (D1199/D1202, user:
//! 'attention is all you need'; attention BETWEEN INPUTS). LEVER ATTENTION v1 on the champion NNUE (812C2118, read
//! from the bundled .tnn as f32):
//!   tokens  = the champion's input groups per perspective (G1 square state + FT bias, G2 anchor, G3 stone ring,
//!             G5 2-ply exchange, G6 global) -> 2 x 5 = 10 token vectors a_t (256 wide = that group's FT rows summed;
//!             the sum of a perspective's 5 tokens IS the champion's accumulator);
//!   attention q_t = Q a_t + eq_t, k_t = K a_t + ek_t (8-dim), alpha = softmax_j(q_i . k_j / sqrt 8) over the 10 tokens,
//!             u_t = U . a_t + ub_t, gamma_i = sum_j alpha_ij u_j;
//!   accumulator' (perspective p) = sum_g (1 + gamma_{p,g}) a_{p,g}  -> the champion's CReLU / l1 / l2 / out / psqt.
//! U, ub start at ZERO, so step 0 is EXACTLY the champion (C14). Trained: Q, K, eq, ek, U, ub + the head (l1, l2, out);
//! the FT rows and psqt stay FROZEN (v1: the head and the attention learn on fixed levers).
//! `--no-attn` = the CONTROL arm: the same run with the attention off (head fine-tune only), so a gain of the attention
//! arm over the control is the attention's, not the recipe's.
//! Lab standard data (whole train_v13, held-out tail, champion target), D4 augmentation always, saturation stop (C15,
//! same rule as jat3-train). Outputs {out}.log, {out}_levatt.bin.
use crate::extval::own_opp;
use crate::hcetrain::{sig, target};
use crate::util::{arg, Rec};
use crate::zattrain::{d4maps, sym};
use std::process::ExitCode;
use titanium::nnfeat::{self, G2_OFF, G3_OFF, G5_OFF, G6_OFF};
use titanium::nnue::{Nnue, BUNDLED_NNUE};
use titanium::Board;

const NG: usize = 5;
const NT: usize = 2 * NG;
const DK: usize = 8;

/// the champion's f32 tensors (parsed exactly like the hub's TrainableNet::from_tnn_bytes)
#[derive(Clone)]
struct Champ {
    w: usize,
    l1: usize,
    l2: usize,
    nb: usize,
    pairwise: bool,
    out_in: usize,
    v2: bool,
    ft: Vec<f32>,
    ftb: Vec<f32>,
    psqt: Vec<f32>,
}

/// trainable parameters, one flat vector: [l1_w][l1_b][l2_w][l2_b][out_w][out_b][Q][K][eq][ek][U][ub]
#[derive(Clone, Copy)]
struct Lay {
    l1w: usize,
    l1b: usize,
    l2w: usize,
    l2b: usize,
    ow: usize,
    ob: usize,
    q: usize,
    k: usize,
    eq: usize,
    ek: usize,
    u: usize,
    ub: usize,
    n: usize,
}

fn parse(d: &[u8]) -> (Champ, Vec<f32>, Lay) {
    let v2 = &d[..4] == b"TNN2";
    let dis = &d[..4] == b"TNN3";
    let u = |i: usize| u32::from_le_bytes(d[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
    let (nf, w, l1, l2, nb) = (u(0), u(1), u(2), u(3), u(4));
    let pairwise = (v2 || dis) && u(6) != 0;
    let din = if pairwise { w } else { 2 * w };
    let out_in = if v2 { l1 + l2 } else { l2 };
    let mut p = if v2 || dis { 32 } else { 28 };
    let mut take = |n: usize| -> Vec<f32> {
        let v = d[p..p + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        p += 4 * n;
        v
    };
    let ft = take(nf * w);
    let ftb = take(w);
    let psqt = take(nf * nb);
    let (l1w, l1b, l2w, l2b, ow, ob) = (take(nb * l1 * din), take(nb * l1), take(nb * l2 * l1), take(nb * l2), take(nb * out_in), take(nb));
    let mut lay = Lay { l1w: 0, l1b: 0, l2w: 0, l2b: 0, ow: 0, ob: 0, q: 0, k: 0, eq: 0, ek: 0, u: 0, ub: 0, n: 0 };
    let mut off = 0usize;
    let mut put = |len: usize| {
        let o = off;
        off += len;
        o
    };
    lay.l1w = put(l1w.len());
    lay.l1b = put(l1b.len());
    lay.l2w = put(l2w.len());
    lay.l2b = put(l2b.len());
    lay.ow = put(ow.len());
    lay.ob = put(ob.len());
    lay.q = put(DK * w);
    lay.k = put(DK * w);
    lay.eq = put(NT * DK);
    lay.ek = put(NT * DK);
    lay.u = put(w);
    lay.ub = put(NT);
    lay.n = off;
    let mut prm = vec![0f32; lay.n];
    prm[lay.l1w..lay.l1b].copy_from_slice(&l1w);
    prm[lay.l1b..lay.l2w].copy_from_slice(&l1b);
    prm[lay.l2w..lay.l2b].copy_from_slice(&l2w);
    prm[lay.l2b..lay.ow].copy_from_slice(&l2b);
    prm[lay.ow..lay.ob].copy_from_slice(&ow);
    prm[lay.ob..lay.q].copy_from_slice(&ob);
    // Q, K, eq, ek: small random (U = 0 keeps the output exact; Q / K get gradients once U moves)
    let mut s = 0x2468_ACE0_1357_9BDFu64;
    let sc = 0.5 / (w as f32).sqrt();
    for x in prm[lay.q..lay.u].iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * sc;
    }
    (Champ { w, l1, l2, nb, pairwise, out_in, v2, ft, ftb, psqt }, prm, lay)
}

fn group_of(f: usize) -> usize {
    if f < G2_OFF {
        0
    } else if f < G3_OFF {
        1
    } else if f < G5_OFF {
        2
    } else if f < G6_OFF {
        3
    } else {
        4
    }
}

/// per-sample workspace
struct Ws {
    a: Vec<f32>,     // NT x w token vectors
    acc: Vec<f32>,   // 2 x w raw accumulator'
    x: Vec<f32>,     // din
    h1r: Vec<f32>,
    h1: Vec<f32>,
    h2r: Vec<f32>,
    h2: Vec<f32>,
    q: [[f32; DK]; NT],
    k: [[f32; DK]; NT],
    al: [[f32; NT]; NT],
    uu: [f32; NT],
    gam: [f32; NT],
    ps: [f32; 2],
    bucket: usize,
    dx: Vec<f32>,
    dacc: Vec<f32>,
}

impl Ws {
    fn new(c: &Champ) -> Ws {
        let din = if c.pairwise { c.w } else { 2 * c.w };
        Ws {
            a: vec![0.0; NT * c.w],
            acc: vec![0.0; 2 * c.w],
            x: vec![0.0; din],
            h1r: vec![0.0; c.l1],
            h1: vec![0.0; c.l1],
            h2r: vec![0.0; c.l2],
            h2: vec![0.0; c.l2],
            q: [[0.0; DK]; NT],
            k: [[0.0; DK]; NT],
            al: [[0.0; NT]; NT],
            uu: [0.0; NT],
            gam: [0.0; NT],
            ps: [0.0; 2],
            bucket: 0,
            dx: vec![0.0; din],
            dacc: vec![0.0; 2 * c.w],
        }
    }
}

/// forward; returns y (sigmoid units, cp = 400 y)
fn fwd(c: &Champ, p: &[f32], ly: Lay, attn: bool, b: &Board, ws: &mut Ws) -> f32 {
    let w = c.w;
    let (us, nu, them, nt) = nnfeat::both(b, nnfeat::ALL_GROUPS);
    let e = b.empty().count_ones() as usize;
    let bk = (e * c.nb / 50).min(c.nb - 1);
    ws.bucket = bk;
    ws.a.iter_mut().for_each(|v| *v = 0.0);
    ws.ps = [0.0; 2];
    for (half, (idx, n)) in [(&us, nu), (&them, nt)].into_iter().enumerate() {
        let t0 = half * NG;
        ws.a[t0 * w..(t0 + 1) * w].copy_from_slice(&c.ftb);
        for &f in &idx[..n] {
            let fi = f as usize;
            let t = t0 + group_of(fi);
            for (v, r) in ws.a[t * w..(t + 1) * w].iter_mut().zip(&c.ft[fi * w..(fi + 1) * w]) {
                *v += *r;
            }
            ws.ps[half] += c.psqt[fi * c.nb + bk];
        }
    }
    let inv = 1.0 / (DK as f32).sqrt();
    if attn {
        for t in 0..NT {
            let at = &ws.a[t * w..(t + 1) * w];
            for d in 0..DK {
                ws.q[t][d] = p[ly.eq + t * DK + d] + p[ly.q + d * w..ly.q + (d + 1) * w].iter().zip(at).map(|(x, y)| x * y).sum::<f32>();
                ws.k[t][d] = p[ly.ek + t * DK + d] + p[ly.k + d * w..ly.k + (d + 1) * w].iter().zip(at).map(|(x, y)| x * y).sum::<f32>();
            }
            ws.uu[t] = p[ly.ub + t] + p[ly.u..ly.u + w].iter().zip(at).map(|(x, y)| x * y).sum::<f32>();
        }
        for i in 0..NT {
            let mut mx = f32::MIN;
            let mut s = [0f32; NT];
            for j in 0..NT {
                s[j] = (0..DK).map(|d| ws.q[i][d] * ws.k[j][d]).sum::<f32>() * inv;
                mx = mx.max(s[j]);
            }
            let mut z = 0f32;
            for j in 0..NT {
                s[j] = (s[j] - mx).exp();
                z += s[j];
            }
            let mut g = 0f32;
            for j in 0..NT {
                ws.al[i][j] = s[j] / z;
                g += ws.al[i][j] * ws.uu[j];
            }
            ws.gam[i] = g;
        }
    } else {
        ws.gam = [0.0; NT];
    }
    for half in 0..2 {
        let acc = &mut ws.acc[half * w..(half + 1) * w];
        acc.iter_mut().for_each(|v| *v = 0.0);
        for g in 0..NG {
            let t = half * NG + g;
            let m = 1.0 + ws.gam[t];
            for (v, x) in acc.iter_mut().zip(&ws.a[t * w..(t + 1) * w]) {
                *v += m * *x;
            }
        }
    }
    let din = ws.x.len();
    if c.pairwise {
        let h = w / 2;
        for half in 0..2 {
            for j in 0..h {
                ws.x[half * h + j] = ws.acc[half * w + j].clamp(0.0, 1.0) * ws.acc[half * w + h + j].clamp(0.0, 1.0);
            }
        }
    } else {
        for i in 0..din {
            ws.x[i] = ws.acc[i].clamp(0.0, 1.0);
        }
    }
    for j in 0..c.l1 {
        let r = bk * c.l1 + j;
        let s: f32 = p[ly.l1w + r * din..ly.l1w + (r + 1) * din].iter().zip(&ws.x).map(|(a, b)| a * b).sum();
        ws.h1r[j] = s + p[ly.l1b + r];
        ws.h1[j] = ws.h1r[j].clamp(0.0, 1.0);
    }
    for j in 0..c.l2 {
        let r = bk * c.l2 + j;
        let s: f32 = p[ly.l2w + r * c.l1..ly.l2w + (r + 1) * c.l1].iter().zip(&ws.h1).map(|(a, b)| a * b).sum();
        ws.h2r[j] = s + p[ly.l2b + r];
        ws.h2[j] = ws.h2r[j].clamp(0.0, 1.0);
    }
    let row = &p[ly.ow + bk * c.out_in..ly.ow + (bk + 1) * c.out_in];
    let o = if c.v2 {
        row[..c.l1].iter().zip(&ws.h1).map(|(a, b)| a * b).sum::<f32>() + row[c.l1..].iter().zip(&ws.h2).map(|(a, b)| a * b).sum::<f32>()
    } else {
        row.iter().zip(&ws.h2).map(|(a, b)| a * b).sum::<f32>()
    } + p[ly.ob + bk];
    o + 0.5 * (ws.ps[0] - ws.ps[1])
}

/// D1200/D1267: Playable wrapper around trained `*_levatt.bin` attention head + champion trunk.
pub struct LevAttNet {
    c: Champ,
    prm: Vec<f32>,
    ly: Lay,
    attn: bool,
}

impl LevAttNet {
    pub fn load(path: &str) -> Self {
        let (c, mut prm, ly) = parse(BUNDLED_NNUE);
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(
            bytes.len(),
            ly.n * 4,
            "{path}: size {} != expected {} bytes",
            bytes.len(),
            ly.n * 4
        );
        for (dst, ch) in prm.iter_mut().zip(bytes.chunks_exact(4)) {
            *dst = f32::from_le_bytes(ch.try_into().unwrap());
        }
        let attn = prm[ly.u..ly.ub + NT].iter().any(|&v| v != 0.0);
        Self { c, prm, ly, attn }
    }

    pub fn eval_f(&self, b: &Board) -> f32 {
        thread_local! {
            static TL_WS: std::cell::RefCell<Option<Ws>> = const { std::cell::RefCell::new(None) };
        }
        TL_WS.with(|cell| {
            let mut slot = cell.borrow_mut();
            let ws = slot.get_or_insert_with(|| Ws::new(&self.c));
            if ws.a.len() != NT * self.c.w {
                *ws = Ws::new(&self.c);
            }
            400.0 * fwd(&self.c, &self.prm, self.ly, self.attn, b, ws)
        })
    }

    pub fn eval(&self, b: &Board) -> i32 {
        self.eval_f(b).round().clamp(-30000.0, 30000.0) as i32
    }
}

/// backward of dy (dL/dy) into the gradient vector g (same layout as the parameters)
fn bwd(c: &Champ, p: &[f32], ly: Lay, attn: bool, ws: &mut Ws, dy: f32, g: &mut [f32]) {
    let (w, l1, l2, bk) = (c.w, c.l1, c.l2, ws.bucket);
    let din = ws.x.len();
    g[ly.ob + bk] += dy;
    let mut dh1 = vec![0f32; l1];
    let mut dh2 = vec![0f32; l2];
    let orow = ly.ow + bk * c.out_in;
    if c.v2 {
        for j in 0..l1 {
            g[orow + j] += dy * ws.h1[j];
            dh1[j] += dy * p[orow + j];
        }
        for j in 0..l2 {
            g[orow + l1 + j] += dy * ws.h2[j];
            dh2[j] += dy * p[orow + l1 + j];
        }
    } else {
        for j in 0..l2 {
            g[orow + j] += dy * ws.h2[j];
            dh2[j] += dy * p[orow + j];
        }
    }
    for j in 0..l2 {
        if ws.h2r[j] <= 0.0 || ws.h2r[j] >= 1.0 {
            continue;
        }
        let r = bk * l2 + j;
        g[ly.l2b + r] += dh2[j];
        for i in 0..l1 {
            g[ly.l2w + r * l1 + i] += dh2[j] * ws.h1[i];
            dh1[i] += dh2[j] * p[ly.l2w + r * l1 + i];
        }
    }
    ws.dx.iter_mut().for_each(|v| *v = 0.0);
    for j in 0..l1 {
        if ws.h1r[j] <= 0.0 || ws.h1r[j] >= 1.0 {
            continue;
        }
        let r = bk * l1 + j;
        g[ly.l1b + r] += dh1[j];
        let base = ly.l1w + r * din;
        for i in 0..din {
            g[base + i] += dh1[j] * ws.x[i];
            ws.dx[i] += dh1[j] * p[base + i];
        }
    }
    if !attn {
        return;
    }
    // CReLU / pairwise -> d accumulator'
    ws.dacc.iter_mut().for_each(|v| *v = 0.0);
    if c.pairwise {
        let h = w / 2;
        for half in 0..2 {
            for j in 0..h {
                let (ra, rb) = (ws.acc[half * w + j], ws.acc[half * w + h + j]);
                let gx = ws.dx[half * h + j];
                if ra > 0.0 && ra < 1.0 {
                    ws.dacc[half * w + j] = gx * rb.clamp(0.0, 1.0);
                }
                if rb > 0.0 && rb < 1.0 {
                    ws.dacc[half * w + h + j] = gx * ra.clamp(0.0, 1.0);
                }
            }
        }
    } else {
        for i in 0..2 * w {
            if ws.acc[i] > 0.0 && ws.acc[i] < 1.0 {
                ws.dacc[i] = ws.dx[i];
            }
        }
    }
    // d gamma_t = d acc'_p . a_t
    let mut dgam = [0f32; NT];
    for t in 0..NT {
        let half = t / NG;
        dgam[t] = ws.dacc[half * w..(half + 1) * w].iter().zip(&ws.a[t * w..(t + 1) * w]).map(|(x, y)| x * y).sum();
    }
    // gamma_i = sum_j al_ij u_j
    let mut du = [0f32; NT];
    let mut dq = [[0f32; DK]; NT];
    let mut dk = [[0f32; DK]; NT];
    let inv = 1.0 / (DK as f32).sqrt();
    for i in 0..NT {
        let mut dal = [0f32; NT];
        let mut dot = 0f32;
        for j in 0..NT {
            du[j] += ws.al[i][j] * dgam[i];
            dal[j] = dgam[i] * ws.uu[j];
            dot += ws.al[i][j] * dal[j];
        }
        for j in 0..NT {
            let ds = ws.al[i][j] * (dal[j] - dot) * inv;
            if ds == 0.0 {
                continue;
            }
            for d in 0..DK {
                dq[i][d] += ds * ws.k[j][d];
                dk[j][d] += ds * ws.q[i][d];
            }
        }
    }
    for t in 0..NT {
        let at = &ws.a[t * w..(t + 1) * w];
        g[ly.ub + t] += du[t];
        if du[t] != 0.0 {
            for (gi, x) in g[ly.u..ly.u + w].iter_mut().zip(at) {
                *gi += du[t] * x;
            }
        }
        for d in 0..DK {
            g[ly.eq + t * DK + d] += dq[t][d];
            g[ly.ek + t * DK + d] += dk[t][d];
            if dq[t][d] != 0.0 {
                for (gi, x) in g[ly.q + d * w..ly.q + (d + 1) * w].iter_mut().zip(at) {
                    *gi += dq[t][d] * x;
                }
            }
            if dk[t][d] != 0.0 {
                for (gi, x) in g[ly.k + d * w..ly.k + (d + 1) * w].iter_mut().zip(at) {
                    *gi += dk[t][d] * x;
                }
            }
        }
    }
}

pub fn run(args: &[String]) -> ExitCode {
    let attn = !args.iter().any(|a| a == "--no-attn");
    let g = |k: &str, d: f64| arg(args, k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let threads = (g("--threads", 7.0) as usize).max(1);
    let bs = g("--bs", 16384.0) as usize;
    let lr0 = g("--lr", 0.0005) as f32;
    for banned in ["--max-epochs", "--epochs", "--steps"] {
        if args.iter().any(|a| a == banned) {
            eprintln!("levatt-train: {banned} is refused (LAWS C15: training stops only by saturation)");
            return ExitCode::from(2);
        }
    }
    let Some(out) = arg(args, "--out") else {
        eprintln!("levatt-train: --out PREFIX is required");
        return ExitCode::from(2);
    };
    if std::path::Path::new(&format!("{out}.log")).exists() && !args.iter().any(|a| a == "--resume") {
        eprintln!("levatt-train: {out}.log exists; refusing to overwrite (or --resume to continue that run)");
        return ExitCode::from(2);
    }
    // D1130 / A13: the standard trains on the whole dataset; --experiment NAME (output under experiments/) is the only
    // way to train on less (stamped EXPERIMENT, never a result); used for the LAWS A14 stop / resume proof (D1310)
    let exp = crate::standard::experiment(args);
    if let Err(e) = crate::standard::check_out(&out, &exp) {
        eprintln!("levatt-train: {e}");
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
                say(crate::standard::stamp(&None::<String>, &st, tr.len(), va.len()), &mut log);
                (tr, va)
            }
            Err(e) => {
                eprintln!("levatt-train: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        let (Some(data), Some(valp)) = (arg(args, "--data"), arg(args, "--val")) else {
            eprintln!("levatt-train --experiment: needs --data X --val V");
            return ExitCode::from(2);
        };
        let mr = g("--max-rows", 1e12) as usize;
        let (tr, va) = (crate::hcetrain::read_rows(&data, mr), crate::hcetrain::read_rows(&valp, 600_000));
        say(format!("EXPERIMENT {}: NOT the standard: {} train rows, {} val rows (data {data})", exp.clone().unwrap_or_default(), tr.len(), va.len()), &mut log);
        (tr, va)
    };
    let (c, mut prm, ly) = parse(BUNDLED_NNUE);
    if args.iter().any(|a| a == "--gradcheck") {
        // analytic vs central finite difference of y on 3 positions, U / ub set non-zero so every path is live
        let mut p = prm.clone();
        let mut s = 0x1111_2222_3333_4444u64;
        for x in p[ly.u..ly.ub + NT].iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2;
        }
        let mut ws = Ws::new(&c);
        let mut worst = 0f64;
        for r in val.iter().step_by(val.len() / 3).take(3) {
            let (me, op) = own_opp(r);
            let b = Board::from_occ(me, op, 0, 0);
            let mut gv = vec![0f32; ly.n];
            fwd(&c, &p, ly, true, &b, &mut ws);
            bwd(&c, &p, ly, true, &mut ws, 1.0, &mut gv);
            for &(name, lo, hi) in &[("l1w", ly.l1w, ly.l1b), ("l2w", ly.l2w, ly.l2b), ("ow", ly.ow, ly.ob), ("Q", ly.q, ly.k), ("K", ly.k, ly.eq), ("eq", ly.eq, ly.ek), ("ek", ly.ek, ly.u), ("U", ly.u, ly.ub), ("ub", ly.ub, ly.n)] {
                // the 4 largest analytic entries of the block + 2 random ones
                let mut idx: Vec<usize> = (lo..hi).collect();
                idx.sort_by(|a, b| gv[*b].abs().partial_cmp(&gv[*a].abs()).unwrap());
                idx.truncate(4);
                for _ in 0..2 {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    idx.push(lo + (s as usize) % (hi - lo));
                }
                for &i in &idx {
                    let h = if matches!(name, "Q" | "K" | "eq" | "ek") { 3e-2f32 } else { 1e-3f32 };
                    let mut pp = p.clone();
                    pp[i] += h;
                    let yp = fwd(&c, &pp, ly, true, &b, &mut ws) as f64;
                    pp[i] -= 2.0 * h;
                    let ym = fwd(&c, &pp, ly, true, &b, &mut ws) as f64;
                    let num = (yp - ym) / (2.0 * h as f64);
                    let ana = gv[i] as f64;
                    let err = (num - ana).abs() / (num.abs() + ana.abs() + 1e-4);
                    worst = worst.max(err);
                    if err > 0.05 {
                        println!("gradcheck {name}[{}]: analytic {ana:.6e} numeric {num:.6e} rel err {err:.3}", i - lo);
                    }
                }
            }
        }
        println!("gradcheck: worst relative error {worst:.4} (f32 + clamp kinks: < ~0.05 = OK)");
        return ExitCode::SUCCESS;
    }
    let maps = d4maps();
    let net = Nnue::bundled();
    // step-0 parity with the engine champion (attention on but U = 0 -> must equal the champion)
    let mut ws0 = Ws::new(&c);
    let mut maxd = 0f32;
    for r in val.iter().step_by((val.len() / 20_000).max(1)) {
        let (me, op) = own_opp(r);
        let b = Board::from_occ(me, op, 0, 0);
        let y = fwd(&c, &prm, ly, attn, &b, &mut ws0) * 400.0;
        maxd = maxd.max((y - net.eval(&b, 0) as f32).abs());
    }
    let vloss = |p: &[f32]| -> f64 {
        let chunk = val.len().div_ceil(threads);
        let parts: Vec<(f64, f64)> = std::thread::scope(|sc| {
            val.chunks(chunk)
                .map(|vs| {
                    let cr = &c;
                    sc.spawn(move || {
                        let mut ws = Ws::new(cr);
                        let (mut l, mut wsum) = (0f64, 0f64);
                        for r in vs {
                            let (me, op) = own_opp(r);
                            let b = Board::from_occ(me, op, 0, 0);
                            let pr = sig(fwd(cr, p, ly, attn, &b, &mut ws));
                            let wt = r.wsum.max(0.0) as f64;
                            l += wt * ((pr - target(r, false)).abs() as f64).powf(2.5);
                            wsum += wt;
                        }
                        (l, wsum)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        parts.iter().map(|p| p.0).sum::<f64>() / parts.iter().map(|p| p.1).sum::<f64>()
    };
    let i_l = vloss(&prm);
    say(
        format!(
            "levatt-train: LEVER ATTENTION v1 ({}) on the champion: {NT} tokens (5 input groups x 2 perspectives, {} wide), {DK}-dim attention -> zero-init group gates; trained {} params (head + attention), FT rows + psqt frozen; {} train rows, {} val rows, {threads} threads, bs {bs}, lr {lr0}, D4 always ({:.0}s)",
            if attn { "ATTENTION ARM" } else { "CONTROL ARM --no-attn: head fine-tune only" },
            c.w,
            ly.n,
            train.len(),
            val.len(),
            t0.elapsed().as_secs_f64()
        ),
        &mut log,
    );
    say(format!("step 0: max |levatt - engine champion| on 20k val positions {maxd:.2} cp (f32 vs quantised engine; must be small); val loss {i_l:.6}"), &mut log);
    let pn = ly.n;
    let mut mo = vec![0f32; pn];
    let mut ve = vec![0f32; pn];
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut astep = 0i32;
    let mut best = (i_l, prm.clone(), 0usize);
    let save = |p: &[f32]| {
        let b: Vec<u8> = p.iter().flat_map(|x| x.to_le_bytes()).collect();
        let _ = crate::stopctl::write_atomic(&format!("{out}_levatt.bin"), &b);
    };
    let mut lr = lr0;
    let (mut bad, mut cycle_gain, mut zero_cycles, mut cycle) = (0usize, false, 0usize, 0usize);
    let mut grads: Vec<Vec<f32>> = (0..threads).map(|_| vec![0f32; pn]).collect();
    let nbatch = train.len().div_ceil(bs);
    // LAWS A14 (user D1307): best weights `{out}_levatt.bin` (atomic, from the start and on every new best), a full
    // checkpoint `{out}.ckpt` after every epoch and on STOP (STOP in the out folder or C:/gitProjects/STOP_ALL, checked
    // before every batch: exact state saved, exit); `--resume` continues exactly. The epoch order is a pure function of
    // the epoch number.
    let ck_path = format!("{out}.ckpt");
    let run_dir = std::path::Path::new(&out).parent().map(|p| if p.as_os_str().is_empty() { std::path::PathBuf::from(".") } else { p.to_path_buf() }).unwrap_or_else(|| ".".into());
    let (mut ep_start, mut b_start, mut tl0, mut tw0) = (1usize, 0usize, 0f64, 0f64);
    if args.iter().any(|a| a == "--resume") {
        let dd = match std::fs::read(&ck_path) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("levatt-train --resume: {ck_path}: {e}");
                return ExitCode::from(2);
            }
        };
        let mut r = crate::stopctl::Rd::new(&dd);
        let ok = (|| -> Result<(), String> {
            if r.str()? != "LCK1" || r.u64()? != pn as u64 {
                return Err("not a levatt-train checkpoint of this net".into());
            }
            ep_start = r.u64()? as usize;
            b_start = r.u64()? as usize;
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
            prm = r.f32s()?;
            mo = r.f32s()?;
            ve = r.f32s()?;
            Ok(())
        })();
        if let Err(e) = ok {
            eprintln!("levatt-train --resume: {ck_path}: {e}");
            return ExitCode::from(2);
        }
        say(format!("RESUMED from {ck_path}: epoch {ep_start} batch {b_start}, best epoch {} = {:.6}", best.2, best.0), &mut log);
    } else {
        if std::path::Path::new(&ck_path).exists() {
            eprintln!("levatt-train: {ck_path} exists: continue it with --resume, or delete it to start over");
            return ExitCode::from(2);
        }
        save(&best.1);
    }
    #[allow(clippy::too_many_arguments)]
    fn save_ck(path: &str, pn: usize, ep: usize, bi: usize, astep: i32, lr: f32, bad: usize, cg: bool, zc: usize, cy: usize, tl: f64, tw: f64, best: &(f64, Vec<f32>, usize), p: &[f32], mo: &[f32], ve: &[f32]) -> Result<(), String> {
        let mut x = crate::stopctl::Wr::default();
        x.str("LCK1").u64(pn as u64).u64(ep as u64).u64(bi as u64).u64(astep as u64).f64(lr as f64).u64(bad as u64).u64(cg as u64).u64(zc as u64).u64(cy as u64).f64(tl).f64(tw).f64(best.0).u64(best.2 as u64).f32s(&best.1).f32s(p).f32s(mo).f32s(ve);
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
                if let Err(e) = save_ck(&ck_path, pn, epoch, bi, astep, lr, bad, cycle_gain, zero_cycles, cycle, tl, tw, &best, &prm, &mo, &ve) {
                    eprintln!("levatt-train: checkpoint: {e}");
                    return ExitCode::from(1);
                }
                say(format!("STOPPED by {f} at epoch {epoch} batch {bi}: state {ck_path}, best (epoch {}) {out}_levatt.bin; the same command + --resume continues", best.2), &mut log);
                return ExitCode::SUCCESS;
            }
            let chunk = batch.len().div_ceil(threads);
            let (pr, trr, mp, cr) = (&prm, &train, &maps, &c);
            let res: Vec<(f64, f64)> = std::thread::scope(|sc| {
                batch
                    .chunks(chunk)
                    .zip(grads.iter_mut())
                    .map(|(idx, gv)| {
                        sc.spawn(move || {
                            let mut ws = Ws::new(cr);
                            let (mut l, mut wsum) = (0f64, 0f64);
                            for &kk in idx {
                                let r = &trr[kk];
                                let wt = r.wsum.max(0.0);
                                if wt == 0.0 {
                                    continue;
                                }
                                let mut x = (kk as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (epoch as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                                x ^= x >> 29;
                                let m = &mp[(x.wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 61) as usize & 7];
                                let (me, op) = own_opp(r);
                                let b = Board::from_occ(sym(me, m), sym(op, m), 0, 0);
                                let y = fwd(cr, pr, ly, attn, &b, &mut ws);
                                let pp = sig(y);
                                let diff = pp - target(r, true);
                                let ad = diff.abs();
                                l += (wt * ad.powf(2.5)) as f64;
                                wsum += wt as f64;
                                let dy = wt * 2.5 * diff.signum() * ad.powf(1.5) * pp * (1.0 - pp);
                                bwd(cr, pr, ly, attn, &mut ws, dy, gv);
                            }
                            (l, wsum)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect()
            });
            let wsum: f64 = res.iter().map(|p| p.1).sum::<f64>().max(1e-9);
            tl += res.iter().map(|p| p.0).sum::<f64>();
            tw += wsum;
            astep += 1;
            let (c1, c2) = (1.0 - b1.powi(astep), 1.0 - b2.powi(astep));
            let inv = 1.0 / wsum as f32;
            for j in 0..pn {
                let mut gi = 0f32;
                for gv in grads.iter_mut() {
                    gi += gv[j];
                    gv[j] = 0.0;
                }
                if gi == 0.0 && mo[j] == 0.0 {
                    continue;
                }
                gi *= inv;
                mo[j] = b1 * mo[j] + (1.0 - b1) * gi;
                ve[j] = b2 * ve[j] + (1.0 - b2) * gi * gi;
                prm[j] -= lr * (mo[j] / c1) / ((ve[j] / c2).sqrt() + eps);
            }
            if (bi + 1) % 500 == 0 {
                let done = (bi + 1) as f64 / nbatch as f64;
                say(format!("  epoch {epoch} batch {}/{nbatch}: train loss {:.6} | {:.0} rows/s | epoch ETA {:.0}s", bi + 1, tl / tw.max(1e-9), ((bi + 1) * bs) as f64 / te.elapsed().as_secs_f64(), te.elapsed().as_secs_f64() * (1.0 / done - 1.0)), &mut log);
            }
        }
        let vl = vloss(&prm);
        if vl < best.0 - 1e-7 {
            best = (vl, prm.clone(), epoch);
            save(&best.1);
            bad = 0;
            cycle_gain = true;
        } else {
            bad += 1;
            lr *= 0.5;
        }
        let gmean = if attn { prm[ly.u..ly.u + c.w].iter().map(|x| x.abs()).sum::<f32>() / c.w as f32 } else { 0.0 };
        say(format!("epoch {epoch}: train loss {:.6} | val loss {vl:.6} | best {:.6} @ {} | lr {lr:.6} | cycle {cycle} | mean |U| {gmean:.2e} | {:.0}s", tl / tw.max(1e-9), best.0, best.2, t0.elapsed().as_secs_f64()), &mut log);
        if bad >= 4 {
            if cycle_gain {
                zero_cycles = 0;
            } else {
                zero_cycles += 1;
            }
            if zero_cycles >= 2 {
                say(format!("SATURATED: two fresh-lr restarts in a row brought no val gain; best epoch {} val loss {:.6} (init = champion {i_l:.6})", best.2, best.0), &mut log);
                break;
            }
            cycle += 1;
            prm.copy_from_slice(&best.1);
            mo.iter_mut().for_each(|v| *v = 0.0);
            ve.iter_mut().for_each(|v| *v = 0.0);
            astep = 0;
            lr = lr0 * 0.5;
            bad = 0;
            cycle_gain = false;
            say(format!("RESTART {cycle}: from the best weights (epoch {}, {:.6}) with fresh lr {lr} and fresh Adam moments", best.2, best.0), &mut log);
        }
        if let Err(e) = save_ck(&ck_path, pn, epoch + 1, 0, astep, lr, bad, cycle_gain, zero_cycles, cycle, 0.0, 0.0, &best, &prm, &mo, &ve) {
            eprintln!("levatt-train: checkpoint: {e}");
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}
