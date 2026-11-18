//! `lab jat3-explain --model X_jat3.bin [--emax 12]` (D1163, user: 'when it is saturated, learn what it is seeing: which
//! relationships / features the champion is missing'). Read-only analysis on the lab-standard held-out rows:
//!   A. where the 3x3 net beats the champion at <= emax empties: loss by situation (empties, mobility, reach-territory);
//!   B. weighted correlations of the champion's error (t - p_C) and of the 3x3 net's correction (p_J - p_C) with simple
//!      endgame quantities (stone difference, reach-territory, parity, mobility);
//!   C. per-window occlusion: mean |change of the 3x3 output| when one window's row is removed (5x5 grid, by band);
//!   D. 3x3 patterns (exact D4-canonical window position + code) where the champion is wrong beyond its mean error at
//!      that empties count, with the 3x3 net's own contribution of that pattern;
//!   E. the positions with the largest gain, as boards.
//! reach-territory = empty squares within distance 2 of only one side's stones.
use crate::extval::own_opp;
use crate::hcetrain::{sig, target};
use crate::jat3::{jidx, row_tab, Jat3Net, NC, NI, NW};
use crate::util::arg;
use std::collections::{BTreeMap, HashMap};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::{Board, MoveList};

struct Row {
    e: u32,
    w: f32,
    t: f32,
    yc: f32,
    yj: f32,
    me: u64,
    op: u64,
    keys: [u32; NW],
    dk: [f32; NW],
}

/// canonical (window, code) key under the 8 board symmetries: min over t of (window' * NC + code')
fn canon_table() -> Vec<u32> {
    let mut canon = vec![u32::MAX; NI];
    let mut pw = [[0u32; 3]; 9];
    for (k, p) in pw.iter_mut().enumerate() {
        for d in 0..3u32 {
            p[d as usize] = d * 3u32.pow(k as u32);
        }
    }
    for t in 0..8 {
        for w in 0..NW {
            let (r, c) = (w / 5, w % 5);
            let ctr = crate::pairs::transform(7 * (r + 1) + c + 1, t);
            let (r2, c2) = (ctr / 7 - 1, ctr % 7 - 1);
            let w2 = 5 * r2 + c2;
            let mut km = [0usize; 9];
            for dr in 0..3 {
                for dc in 0..3 {
                    let s2 = crate::pairs::transform(7 * (r + dr) + c + dc, t);
                    km[3 * dr + dc] = 3 * (s2 / 7 - r2) + (s2 % 7 - c2);
                }
            }
            for code in 0..NC {
                let mut x = code;
                let mut cv = 0u32;
                for &kk in &km {
                    cv += pw[kk][x % 3];
                    x /= 3;
                }
                let key = (w2 * NC) as u32 + cv;
                let e = &mut canon[w * NC + code];
                if key < *e {
                    *e = key;
                }
            }
        }
    }
    canon
}

fn show(key: u32) -> String {
    let (w, mut code) = (key as usize / NC, key as usize % NC);
    let mut s = format!("r{}c{} [", w / 5, w % 5);
    for k in 0..9 {
        s.push(match code % 3 {
            0 => '.',
            1 => 'X',
            _ => 'O',
        });
        code /= 3;
        if k % 3 == 2 && k < 8 {
            s.push('|');
        }
    }
    s.push(']');
    s
}

pub fn explain(args: &[String]) -> ExitCode {
    let Some(mp) = arg(args, "--model") else {
        eprintln!("usage: lab jat3-explain --model X_jat3.bin [--emax 12]");
        return ExitCode::from(2);
    };
    let emax: u32 = arg(args, "--emax").and_then(|v| v.parse().ok()).unwrap_or(12);
    let (st, tr, val) = match crate::standard::standard_split(args, &["--max-rows", "--val", "--data", "--holdout"]) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("jat3-explain: {e}");
            return ExitCode::from(2);
        }
    };
    drop(tr);
    let j = Jat3Net::load(&mp.replace("_jat3.bin", "_jat3.json"));
    let ly = j.ly;
    let hd = ly.h;
    let net = Nnue::bundled();
    println!("{}", crate::standard::stamp(&None, &st, 0, val.len()));
    println!("model {mp} (hidden {hd}); champion = bundled net; analysis band for A/B/D/E: 0-{emax} empties");
    let canon = canon_table();
    let (mut nc0, mut nc6) = (0u64, 0u64);
    for sq in 0..49 {
        if sq % 7 != 0 {
            nc0 |= 1 << sq;
        }
        if sq % 7 != 6 {
            nc6 |= 1 << sq;
        }
    }
    let full = (1u64 << 49) - 1;
    let dil = |b: u64| -> u64 {
        let r = b | ((b & nc6) << 1) | ((b & nc0) >> 1);
        (r | (r << 7) | (r >> 7)) & full
    };
    let head = |h: &[f32]| -> f32 {
        let mut s = j.w[ly.ob()];
        for k in 0..hd {
            let a = h[k].clamp(0.0, 1.0);
            s += a * a * j.w[ly.ow() + k];
        }
        s
    };
    let band = |e: u32| -> usize {
        match e {
            0..=6 => 0,
            7..=12 => 1,
            13..=24 => 2,
            25..=36 => 3,
            _ => 4,
        }
    };
    let bname = ["0-6", "7-12", "13-24", "25-36", "37-49"];
    let mut occ = [[0f64; NW]; 5];
    let mut occw = [0f64; 5];
    let mut rows: Vec<Row> = Vec::new();
    let mut sit: BTreeMap<String, [f64; 4]> = BTreeMap::new();
    let feats = ["safe stones (mine-opp)", "stone diff", "reach-terr (mine-opp)", "projected (diff+terr)", "unreachable empties", "stm mobility", "opp mobility", "parity (empties odd)"];
    let nf = feats.len();
    // per (band 0-6 / 7-emax, feature): sw, sx, sy, sz, sxx, syy, szz, sxy, szy   (x = t - p_C, z = p_J - p_C, y = feature)
    let mut mom = vec![[0f64; 9]; 2 * nf];
    let tab = row_tab();
    let mut ix = [0u32; NW];
    let mut h = vec![0f32; hd];
    let mut h2 = vec![0f32; hd];
    let mut ml = MoveList::new();
    for r in &val {
        let (me, op) = own_opp(r);
        let b = Board::from_occ(me, op, 0, 0);
        let e = b.empty().count_ones();
        let bi = band(e);
        let w = r.wsum.max(0.0);
        let t = target(r, false);
        let yc = net.eval(&b, 0) as f32 / 400.0;
        jidx(me, op, &tab, &mut ix);
        h.copy_from_slice(&j.w[ly.fb()..ly.fb() + hd]);
        for &i in &ix {
            let row = &j.w[i as usize * hd..i as usize * hd + hd];
            for k in 0..hd {
                h[k] += row[k];
            }
        }
        let yj = head(&h);
        let mut dk = [0f32; NW];
        for (wi, &i) in ix.iter().enumerate() {
            let row = &j.w[i as usize * hd..i as usize * hd + hd];
            for k in 0..hd {
                h2[k] = h[k] - row[k];
            }
            dk[wi] = (yj - head(&h2)) * 400.0;
            occ[bi][wi] += w as f64 * dk[wi].abs() as f64;
        }
        occw[bi] += w as f64;
        if e > emax {
            continue;
        }
        let lc = ((sig(yc) - t).abs() as f64).powf(2.5);
        let lj = ((sig(yj) - t).abs() as f64).powf(2.5);
        b.legal_moves_dedup_into(&mut ml);
        let mm = ml.len();
        Board::from_occ(op, me, 0, 0).legal_moves_dedup_into(&mut ml);
        let mo = ml.len();
        let (rm, ro) = (dil(dil(me)), dil(dil(op)));
        let emp = !(me | op) & full;
        let em = (emp & rm & !ro).count_ones() as i32;
        let eo = (emp & ro & !rm).count_ones() as i32;
        let en = (emp & !rm & !ro).count_ones() as i32;
        let d = me.count_ones() as i32 - op.count_ones() as i32;
        // safe = a stone with no empty square in its 8-neighbourhood: no move can land next to it, so it cannot be converted
        // (until a jump vacates a neighbouring origin)
        let nemp = dil(emp);
        let sd = (me & !nemp).count_ones() as i32 - (op & !nemp).count_ones() as i32;
        let mut add = |k: String| {
            let a = sit.entry(k).or_insert([0.0; 4]);
            a[0] += 1.0;
            a[1] += w as f64;
            a[2] += w as f64 * lc;
            a[3] += w as f64 * lj;
        };
        add(format!("a empties {e:02}"));
        add(format!(
            "b mobility {}",
            if mm == 0 && mo == 0 {
                "none (game over)"
            } else if mm == 0 {
                "side to move has NO move"
            } else if mo == 0 {
                "opponent has no move"
            } else {
                "both can move"
            }
        ));
        add(format!("c unreachable empties {}", if en == 0 { "0" } else if en <= 2 { "1-2" } else { "3+" }));
        let td = em - eo;
        add(format!("d reach-terr mine-opp {}", if td <= -3 { "<= -3" } else if td < 0 { "-2..-1" } else if td == 0 { "0" } else if td < 3 { "+1..+2" } else { ">= +3" }));
        let pr = d + td;
        add(format!("e projected diff+terr {}", if pr <= -5 { "<= -5" } else if pr < 0 { "-4..-1" } else if pr == 0 { "0" } else if pr < 5 { "+1..+4" } else { ">= +5" }));
        add(format!("f empties parity {}", if e % 2 == 1 { "odd" } else { "even" }));
        add(format!("g safe stones mine-opp {}", if sd <= -10 { "<= -10" } else if sd < -3 { "-9..-4" } else if sd <= 3 { "-3..+3" } else if sd < 10 { "+4..+9" } else { ">= +10" }));
        let fv = [sd as f64, d as f64, td as f64, pr as f64, en as f64, mm as f64, mo as f64, (e % 2) as f64];
        let bb = if e <= 6 { 0 } else { 1 };
        let (x0, z0, wf) = ((t - sig(yc)) as f64, (sig(yj) - sig(yc)) as f64, w as f64);
        for (fi, &y) in fv.iter().enumerate() {
            let m = &mut mom[bb * nf + fi];
            m[0] += wf;
            m[1] += wf * x0;
            m[2] += wf * y;
            m[3] += wf * z0;
            m[4] += wf * x0 * x0;
            m[5] += wf * y * y;
            m[6] += wf * z0 * z0;
            m[7] += wf * x0 * y;
            m[8] += wf * z0 * y;
        }
        let mut keys = [0u32; NW];
        for (wi, &i) in ix.iter().enumerate() {
            keys[wi] = canon[i as usize];
        }
        rows.push(Row { e, w, t, yc, yj, me, op, keys, dk });
    }

    println!("\nA. WHERE IT WINS ({} rows at <= {emax} empties): weighted loss, champion vs 3x3 net", rows.len());
    println!("{:<42} {:>8} {:>10} {:>10} {:>9} {:>13}", "situation", "rows", "champion", "3x3 net", "3x3 vs C", "share of gain");
    let tot: f64 = sit.iter().filter(|(k, _)| k.starts_with("a ")).map(|(_, a)| a[2] - a[3]).sum();
    let mut last = ' ';
    for (k, a) in &sit {
        let g = k.chars().next().unwrap_or(' ');
        if g != last {
            println!();
            last = g;
        }
        println!("{:<42} {:>8} {:>10.6} {:>10.6} {:>8.2}% {:>12.1}%", &k[2..], a[0] as u64, a[2] / a[1], a[3] / a[1], 100.0 * (a[3] / a[2] - 1.0), 100.0 * (a[2] - a[3]) / tot);
    }

    println!("\nB. WEIGHTED CORRELATIONS: r(champion error t - p_C, feature) | r(3x3 correction p_J - p_C, feature)");
    for (bb, nm) in [(0usize, "0-6".to_string()), (1, format!("7-{emax}"))] {
        println!("  band {nm}:");
        for (fi, f) in feats.iter().enumerate() {
            let m = &mom[bb * nf + fi];
            if m[0] <= 0.0 {
                continue;
            }
            let (ex, ey, ez) = (m[1] / m[0], m[2] / m[0], m[3] / m[0]);
            let (vx, vy, vz) = (m[4] / m[0] - ex * ex, m[5] / m[0] - ey * ey, m[6] / m[0] - ez * ez);
            let rxy = (m[7] / m[0] - ex * ey) / (vx * vy).sqrt().max(1e-12);
            let rzy = (m[8] / m[0] - ez * ey) / (vz * vy).sqrt().max(1e-12);
            println!("    {f:<24} r(err) {rxy:>+7.3}   r(corr) {rzy:>+7.3}   mean {ey:>7.2}");
        }
    }

    println!("\nC. WINDOW OCCLUSION: mean |cp change| of the 3x3 net when one window's row is removed (5x5 grid of window centres)");
    for bi in 0..5 {
        println!("  band {}:", bname[bi]);
        for r in 0..5 {
            let mut s = String::from("   ");
            for c in 0..5 {
                s += &format!("{:>8.1}", occ[bi][5 * r + c] / occw[bi].max(1e-9));
            }
            println!("{s}");
        }
    }

    let mut mb = vec![(0f64, 0f64); 50];
    for r in &rows {
        mb[r.e as usize].0 += r.w as f64 * (r.t - sig(r.yc)) as f64;
        mb[r.e as usize].1 += r.w as f64;
    }
    let mut pat: HashMap<u32, [f64; 6]> = HashMap::new();
    for r in &rows {
        let b0 = (r.t - sig(r.yc)) as f64 - mb[r.e as usize].0 / mb[r.e as usize].1.max(1e-9);
        let cj = (sig(r.yj) - sig(r.yc)) as f64;
        let wf = r.w as f64;
        for wi in 0..NW {
            let a = pat.entry(r.keys[wi]).or_insert([0.0; 6]);
            a[0] += 1.0;
            a[1] += wf;
            a[2] += wf * b0;
            a[3] += wf * b0 * b0;
            a[4] += wf * r.dk[wi] as f64;
            a[5] += wf * cj;
        }
    }
    let mut pv: Vec<(u32, [f64; 6], f64)> = pat.into_iter().filter(|(_, a)| a[0] >= 300.0 && a[1] > 0.0).map(|(k, a)| (k, a, a[2] / a[3].sqrt().max(1e-12))).collect();
    let line = |k: u32, a: &[f64; 6], z: f64| {
        println!(
            "  {:<26} n {:>7}  champion bias {:>+6.2} pts (z {:>+6.1})  3x3 row {:>+7.1} cp  3x3 correction {:>+6.2} pts",
            show(k),
            a[0] as u64,
            100.0 * a[2] / a[1],
            z,
            a[4] / a[1],
            100.0 * a[5] / a[1]
        );
    };
    println!("\nD. PATTERNS where the champion is systematically wrong at <= {emax} empties (bias = t - p_C minus its mean at that empties count; X = side to move, O = opponent, rows read top to bottom; >= 300 rows; {} patterns)", pv.len());
    pv.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!(" champion too PESSIMISTIC for the side to move:");
    for (k, a, z) in pv.iter().take(20) {
        line(*k, a, *z);
    }
    println!(" champion too OPTIMISTIC for the side to move:");
    for (k, a, z) in pv.iter().rev().take(20) {
        line(*k, a, *z);
    }
    pv.sort_by(|a, b| (b.1[4].abs() / b.1[1]).partial_cmp(&(a.1[4].abs() / a.1[1])).unwrap());
    println!(" largest 3x3 row contributions (|cp|):");
    for (k, a, z) in pv.iter().take(20) {
        line(*k, a, *z);
    }

    let mut ex: Vec<(f64, usize)> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.e <= 6)
        .map(|(i, r)| (r.w as f64 * (((sig(r.yc) - r.t).abs() as f64).powf(2.5) - ((sig(r.yj) - r.t).abs() as f64).powf(2.5)), i))
        .collect();
    ex.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\nE. LARGEST GAINS at <= 6 empties (X = side to move):");
    for &(g, i) in ex.iter().take(8) {
        let r = &rows[i];
        println!("  empties {}  target {:.3}  champion {:.3} ({:+.0} cp)  3x3 {:.3} ({:+.0} cp)  gain {:.5}", r.e, r.t, sig(r.yc), r.yc * 400.0, sig(r.yj), r.yj * 400.0, g);
        for rr in 0..7 {
            let mut s = String::from("    ");
            for cc in 0..7 {
                let sq = 7 * rr + cc;
                s.push(if r.me >> sq & 1 == 1 {
                    'X'
                } else if r.op >> sq & 1 == 1 {
                    'O'
                } else {
                    '.'
                });
            }
            println!("{s}");
        }
    }
    ExitCode::SUCCESS
}
