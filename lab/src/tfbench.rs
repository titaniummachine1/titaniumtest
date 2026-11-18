//! `lab tf-bench [--n 20000] [--rounds 5]` (D1196, user: 'would a small transformer be more efficient than a
//! convolution network or a deep NNUE?'): time per eval of a tiny TRANSFORMER over the 49 squares (random weights =
//! arithmetic + memory only, like jat3-bench), single thread, next to the champion NNUE on the same positions.
//! Token = square state (empty / side to move / opponent / blocker) embedding + learned position embedding (d wide);
//! L pre-norm layers of multi-head self-attention (H heads) + MLP (4d, ReLU); mean-pool -> linear -> value.
//! Full recompute every eval (attention has no cheap incremental update). f32, plain loops (the compiler vectorises
//! the inner dot products); an int8 version would be ~2-4x faster, still no incremental path.
//! Measure on a QUIET PC only (D1105).
use crate::util::arg;
use std::process::ExitCode;
use std::time::Instant;
use titanium::nnue::Nnue;
use titanium::Board;

struct Tf {
    d: usize,
    h: usize,
    l: usize,
    emb: Vec<f32>,  // 4 x d
    pos: Vec<f32>,  // 49 x d
    wq: Vec<Vec<f32>>, // per layer d x d
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
    w1: Vec<Vec<f32>>, // d x 4d
    w2: Vec<Vec<f32>>, // 4d x d
    out: Vec<f32>,
    /// D1197: Some = LOCAL attention, square i attends only to these squares (Chebyshev <= 2 = the jump range)
    local: Option<Vec<Vec<usize>>>,
}

/// D1197: attention POOLING: 49 square embeddings (state + position), one learned query, values = x Wv, then a small
/// MLP on the pooled vector. O(49 d) per head instead of O(49^2 d).
struct Pool {
    d: usize,
    emb: Vec<f32>,
    pos: Vec<f32>,
    q: Vec<f32>,
    wv: Vec<f32>,
    w1: Vec<f32>,
    out: Vec<f32>,
}

impl Pool {
    fn new(d: usize) -> Pool {
        let mut s = 0x0F0F_1234_ABCD_9876u64;
        let sc = 1.0 / (d as f32).sqrt();
        let mut m = |n: usize| rnd(&mut s, n, sc);
        Pool { d, emb: m(4 * d), pos: m(49 * d), q: m(d), wv: m(d * d), w1: m(d * 2 * d), out: m(2 * d) }
    }
    fn eval(&self, b: &Board) -> f32 {
        let d = self.d;
        let (me, op, e) = (b.me(), b.opp(), b.empty());
        let mut x = vec![0f32; 49 * d];
        let mut sc = [0f32; 49];
        let mut mx = f32::MIN;
        for sq in 0..49 {
            let t = if (me >> sq) & 1 == 1 { 1 } else if (op >> sq) & 1 == 1 { 2 } else if (e >> sq) & 1 == 1 { 0 } else { 3 };
            let mut s = 0f32;
            for c in 0..d {
                let v = self.emb[t * d + c] + self.pos[sq * d + c];
                x[sq * d + c] = v;
                s += v * self.q[c];
            }
            sc[sq] = s;
            mx = mx.max(s);
        }
        let mut z = 0f32;
        let mut pooled = vec![0f32; d];
        for sq in 0..49 {
            let w = (sc[sq] - mx).exp();
            z += w;
            for c in 0..d {
                pooled[c] += w * x[sq * d + c];
            }
        }
        let mut v = vec![0f32; d];
        Tf::mm(&pooled, &self.wv, 1, d, d, &mut v);
        v.iter_mut().for_each(|a| *a /= z);
        let mut hdn = vec![0f32; 2 * d];
        Tf::mm(&v, &self.w1, 1, d, 2 * d, &mut hdn);
        hdn.iter().zip(&self.out).map(|(a, w)| a.max(0.0) * w).sum()
    }
}

fn rnd(s: &mut u64, n: usize, sc: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * sc
        })
        .collect()
}

impl Tf {
    fn new(d: usize, h: usize, l: usize) -> Tf {
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let sc = 1.0 / (d as f32).sqrt();
        let mut m = |n: usize| rnd(&mut s, n, sc);
        Tf {
            d,
            h,
            l,
            emb: m(4 * d),
            pos: m(49 * d),
            wq: (0..l).map(|_| m(d * d)).collect(),
            wk: (0..l).map(|_| m(d * d)).collect(),
            wv: (0..l).map(|_| m(d * d)).collect(),
            wo: (0..l).map(|_| m(d * d)).collect(),
            w1: (0..l).map(|_| m(d * 4 * d)).collect(),
            w2: (0..l).map(|_| m(4 * d * d)).collect(),
            out: m(d),
            local: None,
        }
    }

    /// local attention: each square attends to the squares within Chebyshev distance 2 (itself included)
    fn local(mut self) -> Tf {
        self.local = Some(
            (0..49usize)
                .map(|i| (0..49usize).filter(|&j| (i / 7).abs_diff(j / 7) <= 2 && (i % 7).abs_diff(j % 7) <= 2).collect())
                .collect(),
        );
        self
    }

    /// y[n x o] = x[n x i] * w[i x o]
    fn mm(x: &[f32], w: &[f32], n: usize, i: usize, o: usize, y: &mut [f32]) {
        for r in 0..n {
            let yr = &mut y[r * o..(r + 1) * o];
            yr.iter_mut().for_each(|v| *v = 0.0);
            for k in 0..i {
                let a = x[r * i + k];
                let wr = &w[k * o..(k + 1) * o];
                for c in 0..o {
                    yr[c] += a * wr[c];
                }
            }
        }
    }

    fn norm(x: &mut [f32], d: usize) {
        for row in x.chunks_mut(d) {
            let m = row.iter().sum::<f32>() / d as f32;
            let v = row.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / d as f32;
            let r = 1.0 / (v + 1e-5).sqrt();
            row.iter_mut().for_each(|a| *a = (*a - m) * r);
        }
    }

    fn eval(&self, b: &Board) -> f32 {
        let (d, h) = (self.d, self.h);
        let hd = d / h;
        let (me, op, e) = (b.me(), b.opp(), b.empty());
        let mut x = vec![0f32; 49 * d];
        for sq in 0..49 {
            let t = if (me >> sq) & 1 == 1 { 1 } else if (op >> sq) & 1 == 1 { 2 } else if (e >> sq) & 1 == 1 { 0 } else { 3 };
            for c in 0..d {
                x[sq * d + c] = self.emb[t * d + c] + self.pos[sq * d + c];
            }
        }
        let (mut xn, mut q, mut k, mut v, mut a, mut o) =
            (vec![0f32; 49 * d], vec![0f32; 49 * d], vec![0f32; 49 * d], vec![0f32; 49 * d], vec![0f32; 49 * d], vec![0f32; 49 * d]);
        let mut hid = vec![0f32; 49 * 4 * d];
        let mut sc = [0f32; 49];
        let scale = 1.0 / (hd as f32).sqrt();
        for li in 0..self.l {
            xn.copy_from_slice(&x);
            Self::norm(&mut xn, d);
            Self::mm(&xn, &self.wq[li], 49, d, d, &mut q);
            Self::mm(&xn, &self.wk[li], 49, d, d, &mut k);
            Self::mm(&xn, &self.wv[li], 49, d, d, &mut v);
            for hh in 0..h {
                let off = hh * hd;
                for i in 0..49 {
                    let all: [usize; 49] = std::array::from_fn(|j| j);
                    let js: &[usize] = match &self.local { Some(l) => &l[i], None => &all };
                    let mut mx = f32::MIN;
                    for (n, &j) in js.iter().enumerate() {
                        let mut s = 0f32;
                        for c in 0..hd {
                            s += q[i * d + off + c] * k[j * d + off + c];
                        }
                        sc[n] = s * scale;
                        mx = mx.max(sc[n]);
                    }
                    let mut z = 0f32;
                    for n in 0..js.len() {
                        sc[n] = (sc[n] - mx).exp();
                        z += sc[n];
                    }
                    for c in 0..hd {
                        let mut s = 0f32;
                        for (n, &j) in js.iter().enumerate() {
                            s += sc[n] * v[j * d + off + c];
                        }
                        a[i * d + off + c] = s / z;
                    }
                }
            }
            Self::mm(&a, &self.wo[li], 49, d, d, &mut o);
            for i in 0..49 * d {
                x[i] += o[i];
            }
            xn.copy_from_slice(&x);
            Self::norm(&mut xn, d);
            Self::mm(&xn, &self.w1[li], 49, d, 4 * d, &mut hid);
            hid.iter_mut().for_each(|t| *t = t.max(0.0));
            Self::mm(&hid, &self.w2[li], 49, 4 * d, d, &mut o);
            for i in 0..49 * d {
                x[i] += o[i];
            }
        }
        let mut y = 0f32;
        for c in 0..d {
            let m: f32 = (0..49).map(|s| x[s * d + c]).sum::<f32>() / 49.0;
            y += m * self.out[c];
        }
        y
    }
}

pub fn run(args: &[String]) -> ExitCode {
    let n: usize = arg(args, "--n").map_or(20000, |v| v.parse().unwrap());
    let rounds: usize = arg(args, "--rounds").map_or(5, |v| v.parse().unwrap());
    // positions: random legal-looking boards from a fixed seed (stones + empties; the timing does not depend on them)
    let mut s = 0xDEAD_BEEF_1234_5678u64;
    let mut boards = Vec::with_capacity(n);
    for _ in 0..n {
        let mut me = 0u64;
        let mut op = 0u64;
        for sq in 0..49 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            match s % 3 {
                1 => me |= 1 << sq,
                2 => op |= 1 << sq,
                _ => {}
            }
        }
        boards.push(Board::from_occ(me, op, 0, 0));
    }
    let time = |f: &dyn Fn(&Board) -> f32| -> f64 {
        let mut best = f64::MAX;
        let mut sink = 0f32;
        for _ in 0..rounds {
            let t = Instant::now();
            for b in &boards {
                sink += f(b);
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e9 / n as f64);
        }
        std::hint::black_box(sink);
        best
    };
    let champ = Nnue::bundled();
    println!("tf-bench: {n} positions, best of {rounds} rounds, single thread (MEASURE ON A QUIET PC, D1105)");
    println!("tf-bench: champion NNUE (full refresh)           {:>9.0} ns/eval", time(&|b| champ.eval(b, 0) as f32));
    for &d in &[16usize, 32, 64] {
        let p = Pool::new(d);
        println!("tf-bench: attention POOLING d {d:>2} (1 query)          {:>9.0} ns/eval", time(&|b| p.eval(b)));
    }
    for &(d, h, l) in &[(16usize, 2usize, 1usize), (16, 2, 2), (32, 4, 1), (32, 4, 2), (64, 4, 2)] {
        let tf = Tf::new(d, h, l);
        let macs = l * (49 * d * d * 4 + 2 * 49 * 49 * d + 2 * 49 * d * 4 * d);
        println!("tf-bench: transformer d {d:>2} heads {h} layers {l}       {:>9.0} ns/eval  (~{:.2} M multiply-adds)", time(&|b| tf.eval(b)), macs as f64 / 1e6);
        let tl = Tf::new(d, h, l).local();
        println!("tf-bench:   same, LOCAL attention (jump range)  {:>9.0} ns/eval", time(&|b| tl.eval(b)));
    }
    ExitCode::SUCCESS
}
