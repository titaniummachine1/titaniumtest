//! `lab deep-see` (D958): what does SEARCH see that the static eval does not?
//!
//!   lab deep-see --tdat X.tdat [--pos 3000] [--n1 5000] [--n2 30000] [--stones 150] [--seed 11] [--threads 7] [--out FILE]
//!
//! 1) POSITION level: static eval vs main search at n1 and n2 nodes on random positions; the search CORRECTION
//!    (search score - static eval) is fitted with the compact hand terms (held-out R2, drop-one, weights): which board
//!    features does the eval mis-price, by how much, in which phase.
//! 2) STONE level: flip every stone of `--stones` positions; value of the stone to its owner by the static eval and by an
//!    n1-node search; the gap (search - static) is tabulated and fitted on local terms; the boards with the biggest gaps
//!    are printed so the patterns can be read by eye.
use crate::hcefit::{box2, class, dot, pgroup, phase, pos_feats, Ridge, CLAMP, HAND, NCL, NH, NPH, PD, PH_NAME};
use crate::seescreen::{est, node_ctx};
use crate::util::{arg, ring1, Rec, REC};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::process::ExitCode;
use titanium::nnue::Nnue;
use titanium::search::{SearchLimits, Searcher};
use titanium::{Board, MoveList};

fn sample(path: &str, n: usize, seed: &mut u64) -> Vec<Board> {
    let mut f = File::open(path).expect("tdat");
    let total = f.metadata().map(|m| m.len() as usize / REC).unwrap_or(0) as u64;
    let mut buf = [0u8; REC];
    let mut ml = MoveList::new();
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        if f.seek(SeekFrom::Start((*seed % total) * REC as u64)).is_err() || f.read_exact(&mut buf).is_err() {
            continue;
        }
        let r = Rec::from_bytes(&buf);
        let (own, opp) = if r.stm == 0 { (r.occ0, r.occ1) } else { (r.occ1, r.occ0) };
        let b = Board::from_occ(own, opp, 0, 0);
        b.legal_moves_dedup_into(&mut ml);
        if ml.len() >= 2 {
            out.push(b);
        }
    }
    out
}

/// search scores (side to move) of every board at a budget (> 0 = nodes, < 0 = fixed depth -n), in parallel, fresh tree
fn search_all(boards: &[Board], budget: i64, threads: usize) -> Vec<f64> {
    let lim = if budget > 0 {
        SearchLimits { time: None, max_nodes: Some(budget as u64), max_depth: 64 }
    } else {
        SearchLimits { time: None, max_nodes: None, max_depth: (-budget) as u8 }
    };
    let mut out = vec![0f64; boards.len()];
    let chunk = boards.len().div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (bs, os) in boards.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let lim = &lim;
            sc.spawn(move || {
                let mut se = Searcher::with_tt_bits(20);
                for (b, o) in bs.iter().zip(os.iter_mut()) {
                    se.reset_for_new_game();
                    let r = se.search(b, &lim);
                    *o = (r.score as f64).clamp(-CLAMP, CLAMP);
                }
            });
        }
    });
    out
}

fn show(b: &Board, mark: usize) -> Vec<String> {
    let (me, op) = (b.me(), b.opp());
    (0..7)
        .map(|r| {
            (0..7)
                .map(|c| {
                    let sq = r * 7 + c;
                    let ch = if (me >> sq) & 1 == 1 {
                        if sq == mark { '#' } else { 'x' }
                    } else if (op >> sq) & 1 == 1 {
                        if sq == mark { '@' } else { 'o' }
                    } else {
                        '.'
                    };
                    format!("{ch} ")
                })
                .collect::<String>()
        })
        .collect()
}

pub fn run(args: &[String]) -> ExitCode {
    let Some(path) = arg(args, "--tdat") else {
        eprintln!("usage: lab deep-see --tdat X.tdat [--pos 3000] [--nodes 100,500,1000,5000] [--stone-nodes 1000] [--stones 150] [--threads 7] [--out FILE]");
        return ExitCode::from(2);
    };
    let g = |k: &str, d: u64| arg(args, k).and_then(|v| v.parse::<u64>().ok()).unwrap_or(d);
    let (npos, nst, threads, ns) = (g("--pos", 3000) as usize, g("--stones", 150) as usize, g("--threads", 7) as usize, g("--stone-nodes", 1000));
    // budgets: plain number = nodes, dN = fixed depth N (user D962)
    let lv_txt: Vec<String> = arg(args, "--nodes").unwrap_or_else(|| "d1,d2,d3,d4,100,1000,5000".into()).split(',').map(|v| v.trim().to_string()).collect();
    let levels: Vec<i64> = lv_txt.iter().map(|v| if let Some(d) = v.strip_prefix('d') { -(d.parse::<i64>().unwrap_or(1)) } else { v.parse().unwrap_or(1000) }).collect();
    let mut seed = g("--seed", 11) | 1;
    let out_path = arg(args, "--out");
    let net = Nnue::bundled();
    let ring = ring1();
    let t0 = std::time::Instant::now();
    let mut rep = String::new();
    let mut say = |s: String| {
        println!("{s}");
        rep.push_str(&s);
        rep.push('\n');
    };

    // ---------------- 1) position level: what does a SHALLOW search change?
    let boards = sample(&path, npos, &mut seed);
    let stat: Vec<f64> = boards.iter().map(|b| (net.eval(b, 0) as f64).clamp(-CLAMP, CLAMP)).collect();
    let sl: Vec<Vec<f64>> = levels.iter().map(|&n| search_all(&boards, n, threads)).collect();
    say(format!("deep-see: {npos} positions, static eval vs main search at {:?} (dN = depth N, else nodes) ({:.0}s)", lv_txt, t0.elapsed().as_secs_f64()));
    say("1) SEARCH CORRECTION per phase and budget: mean |search - static| cp / corr(static, search) / sign flips (|score| > 50)".into());
    for ph in 0..NPH {
        let idx: Vec<usize> = (0..npos).filter(|&i| phase(boards[i].empty()) == ph).collect();
        if idx.len() < 20 {
            continue;
        }
        let n = idx.len() as f64;
        let mut line = format!("  {:<14} n {:>5} |", PH_NAME[ph], idx.len());
        for (li, s) in sl.iter().enumerate() {
            let md = idx.iter().map(|&i| (s[i] - stat[i]).abs()).sum::<f64>() / n;
            let fl = 100.0 * idx.iter().filter(|&&i| s[i].signum() != stat[i].signum() && s[i].abs() > 50.0).count() as f64 / n;
            let (ma, mb) = (idx.iter().map(|&i| stat[i]).sum::<f64>() / n, idx.iter().map(|&i| s[i]).sum::<f64>() / n);
            let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
            for &i in &idx {
                sab += (stat[i] - ma) * (s[i] - mb);
                saa += (stat[i] - ma).powi(2);
                sbb += (s[i] - mb).powi(2);
            }
            line.push_str(&format!(" {}: {:>5.1} / {:.3} / {:>4.1}% |", lv_txt[li], md, sab / (saa * sbb).sqrt(), fl));
        }
        say(line);
    }
    // fit each budget's correction with the compact hand terms (material, weakness count, hand terms; no PST tables)
    let keep: Vec<bool> = (0..PD).map(|i| { let gp = pgroup(i); gp == 0 || gp == 1 || gp == 5 || gp >= 6 }).collect();
    let mut ml = MoveList::new();
    let mut sc = Vec::new();
    let feats: Vec<Vec<(u16, f32)>> = boards
        .iter()
        .map(|b| {
            let mut x = Vec::new();
            pos_feats(b, &ring, &mut ml, &mut x, None, &mut sc);
            x.retain(|&(i, _)| keep[i as usize]);
            x
        })
        .collect();
    let ntr = npos * 3 / 4;
    let fit = |drop: Option<usize>, target: &[f64]| -> (f64, Vec<f64>) {
        let mut r = Ridge::new(PD);
        let k: Vec<bool> = (0..PD).map(|i| keep[i] && Some(pgroup(i)) != drop).collect();
        for i in 0..ntr {
            let x: Vec<(u16, f32)> = feats[i].iter().copied().filter(|&(j, _)| k[j as usize]).collect();
            r.add(&x, target[i]);
        }
        let w = r.solve(&k, 30.0);
        let my = (ntr..npos).map(|i| target[i]).sum::<f64>() / (npos - ntr) as f64;
        let (mut se, mut st) = (0.0, 0.0);
        for i in ntr..npos {
            let x: Vec<(u16, f32)> = feats[i].iter().copied().filter(|&(j, _)| k[j as usize]).collect();
            se += (target[i] - dot(&w, &x)).powi(2);
            st += (target[i] - my).powi(2);
        }
        (1.0 - se / st, w)
    };
    let gname = |gp: usize| match gp { 1 => "material".to_string(), 5 => "weakness count".to_string(), _ => HAND[gp - 6].to_string() };
    for (li, s) in sl.iter().enumerate() {
        // undecided positions only (|static| and |search| < 600): the part a pattern could explain
        let corr: Vec<f64> = (0..npos).map(|i| if stat[i].abs() < 600.0 && s[i].abs() < 600.0 { s[i] - stat[i] } else { 0.0 }).collect();
        let (r2_all, w_all) = fit(None, &corr);
        let mut abl: Vec<(usize, f64)> = (6..6 + NH).chain([1usize, 5]).map(|gp| (gp, r2_all - fit(Some(gp), &corr).0)).collect();
        abl.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        say(format!("  {} correction (undecided positions) explained by hand terms: held-out R2 {:.4}; top terms (dR2, cp per unit by phase late..open):", lv_txt[li], r2_all));
        for (gp, d) in abl.iter().take(5) {
            let ws: Vec<String> = (0..NPH)
                .map(|ph| {
                    let base = ph * crate::hcefit::PB;
                    let ix = (0..crate::hcefit::PB).map(|k| base + k).find(|&i| pgroup(i) == *gp).unwrap_or(base);
                    format!("{:.1}", w_all[ix])
                })
                .collect();
            say(format!("     {:<20} {:>7.4}   {}", gname(*gp), d, ws.join(" / ")));
        }
    }

    // stone level on UNDECIDED positions only, at --stone-nodes
    let sel: Vec<usize> = (0..npos).filter(|&i| stat[i].abs() < 300.0).take(nst).collect();
    let sbv: Vec<Board> = sel.iter().map(|&i| boards[i].clone()).collect();
    let stat: Vec<f64> = sel.iter().map(|&i| stat[i]).collect();
    if nst == 0 {
        if let Some(p) = &out_path {
            let _ = File::create(p).and_then(|mut f| f.write_all(rep.as_bytes()));
        }
        return ExitCode::SUCCESS;
    }
    let ps: Vec<f64> = search_all(&sbv, ns as i64, threads);
    let n1 = ns;
    let sb = &sbv[..];    // ---------------- 2) stone level
    struct St {
        pos: usize,
        sq: usize,
        side: usize,
        ph: usize,
        cl: usize,
        wk: usize,
        on: usize,
        en: usize,
        m: usize,
        bs: usize,
        sv: f64,
        dv: f64,
    }
    let mut flips: Vec<Board> = Vec::new();
    let mut meta: Vec<St> = Vec::new();
    for (pi, b) in sb.iter().enumerate() {
        let (me, op, e) = (b.me(), b.opp(), b.empty());
        let nd = [node_ctx(b), node_ctx(&Board::from_occ(op, me, 0, 0))];
        for sq in 0..49 {
            let bit = 1u64 << sq;
            let side = if me & bit != 0 { 0 } else if op & bit != 0 { 1 } else { continue };
            let (s, o) = if side == 0 { (me, op) } else { (op, me) };
            let lo = box2(o) & e;
            let r2 = box2(bit) & !ring[sq] & !bit;
            let (d1, d2) = (ring[sq] & e, r2 & e);
            let mut bs = -9i32;
            let mut ds = d1 | d2;
            while ds != 0 {
                let d = ds.trailing_zeros() as usize;
                ds &= ds - 1;
                let clone = (d1 >> d) & 1 == 1;
                let my = clone as i32 + 2 * (ring[d] & o).count_ones() as i32;
                let st = est(&nd[side], &ring, sq as u8, d as u8, clone, 0);
                bs = bs.max(my - if st == 0 { -3 } else { st as i32 - 1 });
            }
            let fl = if side == 0 { Board::from_occ(me & !bit, op | bit, 0, 0) } else { Board::from_occ(me | bit, op & !bit, 0, 0) };
            flips.push(fl);
            meta.push(St {
                pos: pi,
                sq,
                side,
                ph: phase(e),
                cl: class(sq),
                wk: (ring[sq] & lo).count_ones() as usize,
                on: (ring[sq] & s).count_ones() as usize,
                en: (ring[sq] & o).count_ones() as usize,
                m: (d1 | d2).count_ones() as usize,
                bs: (bs.clamp(-8, 8) + 8) as usize,
                sv: 0.0,
                dv: 0.0,
            });
        }
    }
    let fs = search_all(&flips, n1 as i64, threads);
    for (k, st) in meta.iter_mut().enumerate() {
        let sg = if st.side == 0 { 1.0 } else { -1.0 };
        st.sv = sg * (stat[st.pos] - (net.eval(&flips[k], 0) as f64).clamp(-CLAMP, CLAMP));
        st.dv = sg * (ps[st.pos] - fs[k]);
    }
    say(String::new());
    say(format!("2) STONE LEVEL: {} stones of {} positions flipped; value to its owner by STATIC eval vs {n1}-node SEARCH ({:.0}s)", meta.len(), sb.len(), t0.elapsed().as_secs_f64()));
    let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
    let gaps: Vec<f64> = meta.iter().map(|s| s.dv - s.sv).collect();
    let absmean = gaps.iter().map(|g| g.abs()).sum::<f64>() / gaps.len() as f64;
    let (ma, mb) = (mean(&meta.iter().map(|s| s.sv).collect::<Vec<_>>()), mean(&meta.iter().map(|s| s.dv).collect::<Vec<_>>()));
    say(format!("  mean stone value: static {:.1} cp, search {:.1} cp; mean |search - static| per stone {:.1} cp", ma, mb, absmean));
    say("  by weakness count 0..8 (own+enemy stones, all phases): static / search / n".into());
    let mut s = String::from("  ");
    for w in 0..9 {
        let v: Vec<&St> = meta.iter().filter(|x| x.wk == w).collect();
        if v.len() >= 30 {
            s.push_str(&format!(" w{w}: {:.0}/{:.0}[{}]", mean(&v.iter().map(|x| x.sv).collect::<Vec<_>>()), mean(&v.iter().map(|x| x.dv).collect::<Vec<_>>()), v.len()));
        }
    }
    say(s);
    say("  by square class 0..9: static / search".into());
    let mut s = String::from("  ");
    for cl in 0..NCL {
        let v: Vec<&St> = meta.iter().filter(|x| x.cl == cl).collect();
        if !v.is_empty() {
            s.push_str(&format!(" c{cl}: {:.0}/{:.0}", mean(&v.iter().map(|x| x.sv).collect::<Vec<_>>()), mean(&v.iter().map(|x| x.dv).collect::<Vec<_>>())));
        }
    }
    say(s);
    say("  by the stone's best SAFE capture (gain - best reply, -8..8): static / search".into());
    let mut s = String::from("  ");
    for b in 0..17 {
        let v: Vec<&St> = meta.iter().filter(|x| x.bs == b).collect();
        if v.len() >= 30 {
            s.push_str(&format!(" {}: {:.0}/{:.0}", b as i32 - 8, mean(&v.iter().map(|x| x.sv).collect::<Vec<_>>()), mean(&v.iter().map(|x| x.dv).collect::<Vec<_>>())));
        }
    }
    say(s);
    // which local terms predict the GAP (what the eval misses per stone)
    let nt = meta.len() * 3 / 4;
    let od = 2 * NPH * (NCL + 9 + 9 + 9 + 25 + 17) + 2 * NPH;
    let fx = |x: &St, gset: u8, v: &mut Vec<(u16, f32)>| {
        v.clear();
        let sp = x.side * NPH + x.ph;
        let mut off = 0usize;
        v.push((sp as u16, 1.0));
        off += 2 * NPH;
        let parts: [(u8, usize, usize); 6] = [(1, NCL, x.cl), (2, 9, x.wk), (4, 9, x.on), (4, 9, x.en), (8, 25, x.m.min(24)), (16, 17, x.bs)];
        for (bit, n, val) in parts {
            if gset & bit != 0 {
                v.push(((off + sp * n + val) as u16, 1.0));
            }
            off += 2 * NPH * n;
        }
    };
    let gfit = |gset: u8| -> f64 {
        let mut r = Ridge::new(od);
        let mut v = Vec::new();
        for (x, gv) in meta[..nt].iter().zip(&gaps[..nt]) {
            fx(x, gset, &mut v);
            r.add(&v, *gv);
        }
        let w = r.solve(&vec![true; od], 5.0);
        let my = mean(&gaps[nt..]);
        let (mut se, mut st) = (0.0, 0.0);
        for (x, gv) in meta[nt..].iter().zip(&gaps[nt..]) {
            fx(x, gset, &mut v);
            se += (gv - dot(&w, &v)).powi(2);
            st += (gv - my).powi(2);
        }
        1.0 - se / st
    };
    say("  how much of the per-stone GAP (search - static) do local terms explain? held-out R2:".into());
    for (name, gs) in [("square class", 1u8), ("+ weaknesses", 3), ("+ 3x3 counts", 7), ("+ piece mobility", 15), ("+ best safe capture", 31), ("best safe capture only", 16)] {
        say(format!("   {:<26} {:.4}", name, gfit(gs)));
    }
    // the boards with the biggest gaps
    let mut order: Vec<usize> = (0..meta.len()).collect();
    order.sort_by(|&a, &b| gaps[b].abs().partial_cmp(&gaps[a].abs()).unwrap());
    say("  biggest gaps (x = side to move, o = opponent, # / @ = the flipped stone):".into());
    let mut shown = Vec::new();
    for &k in order.iter() {
        let x = &meta[k];
        if shown.contains(&x.pos) {
            continue;
        }
        shown.push(x.pos);
        say(format!(
            "   stone {} ({}), weaknesses {}, own/enemy nbrs {}/{}, moves {}, safe capture {}: static {:.0} cp, search {:.0} cp (gap {:+.0}); position static {:.0}, search {:.0}",
            x.sq, if x.side == 0 { "own" } else { "enemy" }, x.wk, x.on, x.en, x.m, x.bs as i32 - 8, x.sv, x.dv, gaps[k], stat[x.pos], ps[x.pos]
        ));
        for line in show(&sb[x.pos], x.sq) {
            say(format!("      {line}"));
        }
        if shown.len() >= 8 {
            break;
        }
    }
    say(format!("done {:.0}s", t0.elapsed().as_secs_f64()));
    if let Some(p) = out_path {
        if let Some(dir) = std::path::Path::new(&p).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match File::create(&p).and_then(|mut f| f.write_all(rep.as_bytes())) {
            Ok(_) => println!("report written: {p}"),
            Err(e) => eprintln!("{p}: {e}"),
        }
    }
    ExitCode::SUCCESS
}
