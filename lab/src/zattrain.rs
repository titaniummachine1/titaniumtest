//! `lab zataxx-train` (D996): Zataxx's net (36 2x2 windows, 2916 -> 256 SCReLU -> 1) trained with EXACTLY the recipe of
//! `lab hce-train` (same rows, target, loss, validation, saturation rule), initialised from Zataxx's own net6.bin (C14).
//!
//!   lab zataxx-train --data train_v10.tdat --val val_v9_clean.tdat --net C:/gitProjects/Zataxx/src/net6.bin
//!                    [--threads 7] [--bs 16384] [--lr 0.001] [--max-rows N (smoke only)] [--out PREFIX]
//! Online D4 augmentation (random board symmetry per training row per epoch) is always on, as in the NNUE trainer (C12).
//!
//! Float form of Zataxx's quantised forward: h = fb + sum of 36 rows (all / QA), a = clamp(h, 0, 1)^2,
//! out = sum a * ow (/ QB) + ob (/ QA*QB); out is cp / 400 (Zataxx SCALE 400) = the sigmoid input of the hce-train loss.
//! The best single scale on val (D995) is folded into the output layer at init, so step 0 = the calibrated Zataxx net.
//! The best epoch is exported back to Zataxx's i16 format ({out}_net.bin) and the QUANTISED net is re-scored.
use crate::extval::{own_opp, Zat};
use crate::hcetrain::{read_rows, sig, target};
use crate::util::{arg, Rec};
use std::io::Write;
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

const H: usize = 256;
const NI: usize = 2916;
const P: usize = NI * H + 2 * H + 1;
const FB: usize = NI * H;
const OW: usize = FB + H;
const OB: usize = OW + H;
/// D1021: largest float output weight whose quantised value (x64) keeps clamp(h,0,255)*w inside i16 (Zataxx AVX2 path)
const OW_MAX: f32 = 127.0 / 64.0;

fn zidx(us: u64, them: u64, out: &mut [usize; 36]) {
    const POW: [usize; 4] = [1, 3, 9, 27];
    const MASK: u64 = 0b0001_1000_0011;
    for i in 0..6 {
        for j in 0..6 {
            let off = 7 * i + j;
            let mut idx = 81 * (6 * i + j);
            let (mut u, mut t) = ((us >> off) & MASK, (them >> off) & MASK);
            while u > 0 {
                let mut sq = u.trailing_zeros() as usize;
                u &= u - 1;
                if sq > 6 {
                    sq -= 5;
                }
                idx += POW[sq];
            }
            while t > 0 {
                let mut sq = t.trailing_zeros() as usize;
                t &= t - 1;
                if sq > 6 {
                    sq -= 5;
                }
                idx += 2 * POW[sq];
            }
            out[6 * i + j] = idx;
        }
    }
}

/// D1137: the Zataxx 2x2 window indices for jat3's step-0 parity check
pub(crate) fn zidx_pub(us: u64, them: u64, out: &mut [usize; 36]) {
    zidx(us, them, out)
}

/// forward; fills h (pre-activation) and returns out (= cp / 400)
fn fwd(w: &[f32], ix: &[usize; 36], h: &mut [f32; H]) -> f32 {
    h.copy_from_slice(&w[FB..FB + H]);
    for &i in ix {
        let row = &w[i * H..i * H + H];
        for k in 0..H {
            h[k] += row[k];
        }
    }
    let ow = &w[OW..OW + H];
    let mut s = w[OB];
    for k in 0..H {
        let a = h[k].clamp(0.0, 1.0);
        s += a * a * ow[k];
    }
    s
}

/// the 8 board symmetries as square maps (pairs::transform, same as gataxx / the NNUE trainer's D4 set)
pub(crate) fn d4maps() -> [[u8; 49]; 8] {
    let mut m = [[0u8; 49]; 8];
    for (t, mt) in m.iter_mut().enumerate() {
        for (sq, v) in mt.iter_mut().enumerate() {
            *v = crate::pairs::transform(sq, t) as u8;
        }
    }
    m
}

pub(crate) fn sym(mut b: u64, m: &[u8; 49]) -> u64 {
    let mut o = 0u64;
    while b != 0 {
        o |= 1u64 << m[b.trailing_zeros() as usize];
        b &= b - 1;
    }
    o
}

fn rec_ix(r: &Rec, ix: &mut [usize; 36]) {
    let (me, op) = own_opp(r);
    zidx(me, op, ix);
}

pub fn run(args: &[String]) -> ExitCode {
    // D1130: standard = WHOLE dataset (lab/STANDARD_TRAIN.txt) + its held-out tail; --data / --val / --max-rows only with
    // --experiment NAME (output under experiments/, stamped)
    let exp = crate::standard::experiment(args);
    let Some(np) = arg(args, "--net") else {
        eprintln!("usage: lab zataxx-train --net net6.bin [--threads 7] [--bs 16384] [--lr 0.001] [--out PREFIX]  (experiment: --experiment NAME --data X --val V [--max-rows N])");
        return ExitCode::from(2);
    };
    let g = |k: &str, d: f64| arg(args, k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let threads = g("--threads", 7.0) as usize;
    let bs = g("--bs", 16384.0) as usize;
    let mut lr = g("--lr", 0.001) as f32;
    let max_rows = g("--max-rows", 1e12) as usize;
    // online D4 augmentation is ALWAYS on, no switch (user D1001; D999: without it the net overfits); validation unaugmented
    if arg(args, "--d4").is_some() {
        eprintln!("--d4 was removed: D4 augmentation is always on (D1001)");
        return ExitCode::from(2);
    }
    let maps = d4maps();
    let passes = g("--sym-per-epoch", 1.0) as usize;
    if passes != 1 && passes != 8 {
        eprintln!("--sym-per-epoch must be 1 (one random symmetry per row per epoch) or 8 (all 8 per epoch)");
        return ExitCode::from(2);
    }
    let out = arg(args, "--out").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/logs/hce/d996_zattrain".into());
    let t0 = std::time::Instant::now();
    // LAWS A14: a --resume keeps the earlier log and appends to it
    let mut log = if args.iter().any(|a| a == "--resume") { std::fs::read_to_string(format!("{out}.log")).unwrap_or_default() } else { String::new() };
    let say = |s: String, log: &mut String| {
        println!("{s}");
        log.push_str(&s);
        log.push('\n');
        let _ = std::fs::write(format!("{out}.log"), log.as_bytes());
    };
    if let Err(e) = crate::standard::check_out(&out, &exp) {
        eprintln!("zataxx-train: {e}");
        return ExitCode::from(2);
    }
    let (train, val) = if exp.is_none() {
        match crate::standard::standard_split(args, &["--max-rows", "--val"]) {
            Ok((st, tr, va)) => {
                say(crate::standard::stamp(&exp, &st, tr.len(), va.len()), &mut log);
                (tr, va)
            }
            Err(e) => {
                eprintln!("zataxx-train: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        let (Some(data), Some(valp)) = (arg(args, "--data"), arg(args, "--val")) else {
            eprintln!("zataxx-train --experiment: needs --data X --val V");
            return ExitCode::from(2);
        };
        let (tr, va) = (read_rows(&data, max_rows), read_rows(&valp, 600_000));
        say(format!("EXPERIMENT {}: NOT the standard: {} train rows, {} val rows (data {data})", exp.clone().unwrap_or_default(), tr.len(), va.len()), &mut log);
        (tr, va)
    };
    let vix: Vec<[usize; 36]> = val.iter().map(|r| { let mut a = [0; 36]; rec_ix(r, &mut a); a }).collect();

    // init: dequantised net6.bin
    let z = Zat::load(&np);
    let mut w = vec![0f32; P];
    for i in 0..NI * H {
        w[i] = z.fw[i] as f32 / 255.0;
    }
    for k in 0..H {
        w[FB + k] = z.fb[k] as f32 / 255.0;
        w[OW + k] = z.ow[k] as f32 / 64.0;
    }
    w[OB] = z.obias as f32 / (255.0 * 64.0);

    let vloss = |w: &[f32], k: f32| -> f64 {
        let chunk = val.len().div_ceil(threads);
        let parts: Vec<(f64, f64)> = std::thread::scope(|sc| {
            val.chunks(chunk)
                .zip(vix.chunks(chunk))
                .map(|(vs, xs)| {
                    sc.spawn(move || {
                        let mut h = [0f32; H];
                        let (mut l, mut ws) = (0f64, 0f64);
                        for (r, ix) in vs.iter().zip(xs) {
                            let p = sig(k * fwd(w, ix, &mut h));
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
    let raw = vloss(&w, 1.0);
    let mut kb = (raw, 1.0f32);
    let mut k = 0.3f32;
    while k <= 1.5 {
        let l = vloss(&w, k);
        if l < kb.0 {
            kb = (l, k);
        }
        k += 0.01;
    }
    for i in OW..P {
        w[i] *= kb.1;
    }
    for x in &mut w[OW..OB] {
        *x = x.clamp(-OW_MAX, OW_MAX); // D1021 AVX2 bound
    }
    let i_l = vloss(&w, 1.0);
    say(format!("zataxx-train: {} train rows, {} val rows, {P} weights, {threads} threads, bs {bs}, lr {lr}, D4 augmentation always on, {passes} symmetries per row per epoch ({:.0}s)", train.len(), val.len(), t0.elapsed().as_secs_f64()), &mut log);
    say(format!("init = net6.bin dequantised: val loss {raw:.6} raw, best scale {:.2} folded into the output layer -> {i_l:.6} (NNUE 0.008978, D994 HCE+pairs 0.010904)", kb.1), &mut log);

    let (mut mo, mut ve) = (vec![0f32; P], vec![0f32; P]);
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut step = 0i32;
    let mut best = (i_l, w.clone(), 0usize);
    let mut bad = 0usize;
    let mut grads: Vec<Vec<f32>> = (0..threads).map(|_| vec![0f32; P]).collect();
    let cs = P.div_ceil(threads);
    // LAWS A14 (user D1307): best float weights `{out}_best.f32` (atomic, from the start and on every new best), full
    // checkpoint `{out}.ckpt` after every epoch, STOP in the out folder or C:/gitProjects/STOP_ALL checked before every
    // batch (exact state saved, exit), `--resume` continues. Each pass order is a pure function of (epoch, pass).
    let best_path = format!("{out}_best.f32");
    let ck_path = format!("{out}.ckpt");
    let run_dir = std::path::Path::new(&out).parent().map(|p| if p.as_os_str().is_empty() { std::path::PathBuf::from(".") } else { p.to_path_buf() }).unwrap_or_else(|| ".".into());
    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let (mut ep_start, mut pass_start, mut b_start, mut tl0, mut tw0) = (1usize, 0usize, 0usize, 0f64, 0f64);
    if args.iter().any(|a| a == "--resume") {
        let dd = match std::fs::read(&ck_path) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("zataxx-train --resume: {ck_path}: {e}");
                return ExitCode::from(2);
            }
        };
        let mut r = crate::stopctl::Rd::new(&dd);
        let ok = (|| -> Result<(), String> {
            if r.str()? != "ZCK1" || r.u64()? != P as u64 || r.u64()? != passes as u64 {
                return Err("not a zataxx-train checkpoint of this setup".into());
            }
            ep_start = r.u64()? as usize;
            pass_start = r.u64()? as usize;
            b_start = r.u64()? as usize;
            step = r.u64()? as i32;
            lr = r.f64()? as f32;
            bad = r.u64()? as usize;
            tl0 = r.f64()?;
            tw0 = r.f64()?;
            best.0 = r.f64()?;
            best.2 = r.u64()? as usize;
            best.1 = r.f32s()?;
            w = r.f32s()?;
            mo = r.f32s()?;
            ve = r.f32s()?;
            Ok(())
        })();
        if let Err(e) = ok {
            eprintln!("zataxx-train --resume: {ck_path}: {e}");
            return ExitCode::from(2);
        }
        say(format!("RESUMED from {ck_path}: epoch {ep_start} pass {pass_start} batch {b_start}, best epoch {} = {:.6}", best.2, best.0), &mut log);
    } else {
        if std::path::Path::new(&ck_path).exists() {
            eprintln!("zataxx-train: {ck_path} exists: continue it with --resume, or delete it to start over");
            return ExitCode::from(2);
        }
        let _ = crate::stopctl::write_atomic(&best_path, &f32_bytes(&best.1));
    }
    #[allow(clippy::too_many_arguments)]
    fn save_ck(path: &str, passes: usize, ep: usize, pass: usize, bi: usize, step: i32, lr: f32, bad: usize, tl: f64, tw: f64, best: &(f64, Vec<f32>, usize), w: &[f32], mo: &[f32], ve: &[f32]) -> Result<(), String> {
        let mut x = crate::stopctl::Wr::default();
        x.str("ZCK1").u64(P as u64).u64(passes as u64).u64(ep as u64).u64(pass as u64).u64(bi as u64).u64(step as u64).f64(lr as f64).u64(bad as u64).f64(tl).f64(tw).f64(best.0).u64(best.2 as u64).f32s(&best.1).f32s(w).f32s(mo).f32s(ve);
        crate::stopctl::write_atomic(path, &x.0)
    }
    for epoch in ep_start.. {
        let (mut tl, mut tw) = if epoch == ep_start { (tl0, tw0) } else { (0f64, 0f64) };
        // --sym-per-epoch 8 (D1005): one epoch = 8 shuffled passes, row k gets symmetry (s_k + pass) & 7, i.e. every row in
        // all 8 orientations per epoch; validation (and the lr-halving / saturation rule) once per epoch
        let p_first = if epoch == ep_start { pass_start } else { 0 };
        for pass in p_first..passes {
        let mut order: Vec<usize> = (0..train.len()).collect();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ ((epoch * 8 + pass) as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
        for i in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        let first = if epoch == ep_start && pass == p_first { b_start } else { 0 };
        for (bi, batch) in order.chunks(bs).enumerate().skip(first) {
            if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
                if let Err(e) = save_ck(&ck_path, passes, epoch, pass, bi, step, lr, bad, tl, tw, &best, &w, &mo, &ve) {
                    eprintln!("zataxx-train: checkpoint: {e}");
                    return ExitCode::from(1);
                }
                say(format!("STOPPED by {f} at epoch {epoch} pass {pass} batch {bi}: state {ck_path}, best (epoch {}) {best_path}; the same command + --resume continues", best.2), &mut log);
                return ExitCode::SUCCESS;
            }
            let chunk = batch.len().div_ceil(threads);
            let wr = &w;
            let trr = &train;
            let mp = &maps;
            let ps = pass;
            let res: Vec<(f64, f64)> = std::thread::scope(|sc| {
                batch
                    .chunks(chunk)
                    .zip(grads.iter_mut())
                    .map(|(idx, gr)| {
                        sc.spawn(move || {
                            let mut h = [0f32; H];
                            let mut gh = [0f32; H];
                            let mut ix = [0usize; 36];
                            let (mut l, mut ws) = (0f64, 0f64);
                            for &kk in idx {
                                let r = &trr[kk];
                                let wt = r.wsum.max(0.0);
                                if wt == 0.0 {
                                    continue;
                                }
                                // online D4 augmentation, ALWAYS on (user D1001): random symmetry per row per epoch
                                let mut x = (kk as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (epoch as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                                x ^= x >> 29;
                                let m = &mp[((x.wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 61) as usize + ps) & 7];
                                let (me, op) = own_opp(r);
                                zidx(sym(me, m), sym(op, m), &mut ix);
                                let y = fwd(wr, &ix, &mut h);
                                let p = sig(y);
                                let diff = p - target(r, true);
                                let ad = diff.abs();
                                l += (wt * ad.powf(2.5)) as f64;
                                ws += wt as f64;
                                let dy = wt * 2.5 * diff.signum() * ad.powf(1.5) * p * (1.0 - p);
                                gr[OB] += dy;
                                for kk2 in 0..H {
                                    let hk = h[kk2];
                                    let a = hk.clamp(0.0, 1.0);
                                    gr[OW + kk2] += dy * a * a;
                                    gh[kk2] = if hk > 0.0 && hk < 1.0 { dy * wr[OW + kk2] * 2.0 * hk } else { 0.0 };
                                    gr[FB + kk2] += gh[kk2];
                                }
                                for &i in &ix {
                                    let row = &mut gr[i * H..i * H + H];
                                    for kk2 in 0..H {
                                        row[kk2] += gh[kk2];
                                    }
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
            let (c1, c2) = (1.0 - b1.powi(step), 1.0 - b2.powi(step));
            let inv = 1.0 / ws as f32;
            // reduction + Adam, parallel over parameter chunks; zeroes the per-thread gradients
            let mut per_c: Vec<Vec<&mut [f32]>> = (0..threads).map(|_| Vec::new()).collect();
            for gv in grads.iter_mut() {
                for (c, s) in gv.chunks_mut(cs).enumerate() {
                    per_c[c].push(s);
                }
            }
            let lrv = lr;
            std::thread::scope(|sc| {
                for (((gs, wc), mc), vc) in per_c.into_iter().zip(w.chunks_mut(cs)).zip(mo.chunks_mut(cs)).zip(ve.chunks_mut(cs)) {
                    sc.spawn(move || {
                        let mut gs = gs;
                        for j in 0..wc.len() {
                            let mut gi = 0f32;
                            for s in gs.iter_mut() {
                                gi += s[j];
                                s[j] = 0.0;
                            }
                            if gi == 0.0 && mc[j] == 0.0 {
                                continue;
                            }
                            gi *= inv;
                            mc[j] = b1 * mc[j] + (1.0 - b1) * gi;
                            vc[j] = b2 * vc[j] + (1.0 - b2) * gi * gi;
                            wc[j] -= lrv * (mc[j] / c1) / ((vc[j] / c2).sqrt() + eps);
                        }
                    });
                }
            });
            // D1021: Zataxx's AVX2 flatten keeps 16 bits of clamp(h,0,255)*w, so |quantised ow| must stay <= 127
            for x in &mut w[OW..OB] {
                *x = x.clamp(-OW_MAX, OW_MAX);
            }
        }
        }
        let vl = vloss(&w, 1.0);
        if vl < best.0 - 1e-7 {
            best = (vl, w.clone(), epoch);
            bad = 0;
            let _ = crate::stopctl::write_atomic(&best_path, &f32_bytes(&best.1));
        } else {
            bad += 1;
            lr *= 0.5;
        }
        if let Err(e) = save_ck(&ck_path, passes, epoch + 1, 0, 0, step, lr, bad, 0.0, 0.0, &best, &w, &mo, &ve) {
            eprintln!("zataxx-train: checkpoint: {e}");
            return ExitCode::from(1);
        }
        say(format!(
            "epoch {epoch}: train loss {:.6} | val loss {vl:.6} | best {:.6} @ {} | lr {lr:.6} | NNUE 0.008978 | {:.0}s",
            tl / tw.max(1e-9), best.0, best.2, t0.elapsed().as_secs_f64()
        ), &mut log);
        if bad >= 4 {
            say(format!("SATURATED: 4 epochs without improvement; best epoch {} val loss {:.6} (init {i_l:.6})", best.2, best.0), &mut log);
            break;
        }
    }

    // export to Zataxx's i16 format and re-score the QUANTISED net
    let bw = &best.1;
    let q = |x: f32, s: f32| -> i16 { (x * s).round().clamp(-32768.0, 32767.0) as i16 };
    let mut clipped = 0usize;
    let mut bytes = Vec::with_capacity(P * 2);
    for i in 0..P {
        let s = if i < OW { 255.0 } else if i < OB { 64.0 } else { 255.0 * 64.0 };
        let v = bw[i] * s;
        let lim = if (OW..OB).contains(&i) { 127.0 } else { 32767.0 };
        clipped += (v.abs() > lim) as usize;
        let qi = q(bw[i], s).clamp(-(lim as i16), lim as i16);
        bytes.extend_from_slice(&qi.to_le_bytes());
    }
    let np2 = format!("{out}_net.bin");
    let _ = crate::stopctl::write_atomic(&np2, &bytes);
    let mut zq = Zat::load(&np2);
    zq.trunc = true; // score exactly what Zataxx's AVX2 build computes (D1021)
    let net = Nnue::bundled();
    let (mut l, mut ws) = (0f64, 0f64);
    for r in &val {
        let (me, op) = own_opp(r);
        let p = sig(zq.eval(me, op) as f32 / 400.0);
        let wt = r.wsum.max(0.0) as f64;
        l += wt * ((p - target(r, false)).abs() as f64).powf(2.5);
        ws += wt;
    }
    let agree = |f: &dyn Fn(&Board) -> f32| -> f64 {
        let (mut ag, mut n) = (0f64, 0f64);
        let mut ml = MoveList::new();
        for r in val.iter().step_by((val.len() / 2000).max(1)).take(2000) {
            let (me, op) = own_opp(r);
            let b = Board::from_occ(me, op, 0, 0);
            b.legal_moves_dedup_into(&mut ml);
            if ml.len() < 2 {
                continue;
            }
            let (mut bn, mut bz) = ((f32::MIN, 0usize), (f32::MIN, 0usize));
            for (i, mv) in ml.iter().enumerate() {
                let c = b.make(mv);
                let yn = -(net.eval(&c, 0) as f32);
                let yz = -f(&c);
                if yn > bn.0 {
                    bn = (yn, i);
                }
                if yz > bz.0 {
                    bz = (yz, i);
                }
            }
            ag += (bn.1 == bz.1) as u8 as f64;
            n += 1.0;
        }
        100.0 * ag / n
    };
    let a0 = agree(&|c: &Board| z.eval(c.me(), c.opp()) as f32);
    let a1 = agree(&|c: &Board| zq.eval(c.me(), c.opp()) as f32);
    say(format!(
        "exported {np2} ({clipped} weights clipped: output weights to +-127, rest to i16): QUANTISED val loss (AVX2 arithmetic) {:.6} (float best {:.6}); sibling best-move agreement with the NNUE: net6.bin {a0:.2}%, trained {a1:.2}%",
        l / ws, best.0
    ), &mut log);
    ExitCode::SUCCESS
}
