//! `ti spsa`: SPSA tuning of the engine's run-time tunables (D284/D286/D289: the tuner lives in the hub, never in a script).
//!
//! Each iteration k perturbs the whole parameter vector at once: delta_i = +-1, theta+ = theta + c_k delta, theta- = theta - c_k delta
//! (rounded, clamped). The two settings play a paired, seat-balanced, book-driven match through the hub's own SPSA-free gate
//! runner (`ti sprt`, run as a child process, so the game rules, UAI driver and pairing are exactly the gate's). The match score s
//! gives R = 2s - 1 in [-1, 1] and the update is theta_i += a_k R delta_i, the standard fishtest rule.
//!
//! The engine must be the tunable build (exp/tune): `<exe> tune-list` prints "name default" for every knob and `--tune a=1,b=2`
//! sets them. Nodes per move alternate through `--nodes 5000,50000` (user 2026-11-01: never below 5k).
//!
//!   ti spsa --exe titanium-cli.exe --net nnue_main.tnn --book book50k_std5k.txt --knobs knobs.txt --out run.log \
//!           --nodes 5000,50000 --pairs 50 --iters 200 --threads 7 [--book0 60000] [--seed 1] [--resume]
//!           [--warm-nodes 1000 --warm-iters 100]   (user 2026-11-02: first iterations at shallow, fast games, then the --nodes cycle)
//!
//! knobs.txt: one knob per line, `name lo hi c [a]` (c = perturbation size in knob units, a = step size, default c). `#` comments.
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, ExitCode};

struct Knob {
    name: String,
    lo: f64,
    hi: f64,
    c: f64,
    a: f64,
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn rng_next(s: &mut u64) -> u64 {
    // xorshift64*
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    s.wrapping_mul(0x2545F4914F6CDD1D)
}

fn load_knobs(path: &str) -> Result<Vec<Knob>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut v = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 {
            return Err(format!("{path}:{}: expected `name lo hi c [a]`", ln + 1));
        }
        let num = |i: usize| f[i].parse::<f64>().map_err(|_| format!("{path}:{}: bad number '{}'", ln + 1, f[i]));
        let c = num(3)?;
        let a = if f.len() > 4 { num(4)? } else { c };
        v.push(Knob { name: f[0].to_string(), lo: num(1)?, hi: num(2)?, c, a });
    }
    if v.is_empty() {
        return Err(format!("{path}: no knobs"));
    }
    Ok(v)
}

/// "name default" lines printed by `<exe> tune-list`.
fn engine_defaults(exe: &str) -> Result<HashMap<String, f64>, String> {
    let o = Command::new(exe).arg("tune-list").output().map_err(|e| format!("{exe} tune-list: {e}"))?;
    if !o.status.success() {
        return Err(format!("{exe} tune-list failed (not the exp/tune build?)"));
    }
    let mut m = HashMap::new();
    for l in String::from_utf8_lossy(&o.stdout).lines() {
        let mut it = l.split_whitespace();
        if let (Some(n), Some(d)) = (it.next(), it.next().and_then(|d| d.parse::<f64>().ok())) {
            m.insert(n.to_string(), d);
        }
    }
    Ok(m)
}

fn tune_arg(knobs: &[Knob], theta: &[f64]) -> String {
    knobs.iter().zip(theta).map(|(k, t)| format!("{}={}", k.name, t.round() as i64)).collect::<Vec<_>>().join(",")
}

/// (wins, losses, draws) of the candidate from the last `[Pair ...] +w -l =d (` line of the sprt output.
fn parse_wld(out: &str) -> Option<(u32, u32, u32)> {
    let line = out.lines().rev().find(|l| l.starts_with("[Pair"))?;
    let rest = line.split("s] ").nth(1)?;
    let head: String = rest.split(" (").next()?.chars().filter(|c| !c.is_whitespace()).collect();
    let head = head.strip_prefix('+')?;
    let (w, rest) = head.split_once('-')?;
    let (l, d) = rest.split_once('=')?;
    Some((w.parse().ok()?, l.parse().ok()?, d.parse().ok()?))
}

fn last_theta(out_path: &str, knobs: &[Knob]) -> Option<(usize, Vec<f64>)> {
    let text = fs::read_to_string(out_path).ok()?;
    let line = text.lines().rev().find(|l| l.contains("theta_next="))?;
    let k: usize = line.split_whitespace().find_map(|t| t.strip_prefix("k="))?.parse().ok()?;
    let tn = line.split("theta_next=").nth(1)?;
    let mut th = vec![f64::NAN; knobs.len()];
    for kv in tn.split(',') {
        let (n, v) = kv.split_once('=')?;
        if let Some(i) = knobs.iter().position(|kn| kn.name == n.trim()) {
            th[i] = v.trim().parse().ok()?;
        }
    }
    if th.iter().any(|t| t.is_nan()) {
        return None;
    }
    Some((k + 1, th))
}

pub fn run_spsa(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("spsa: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let need = |n: &str| flag(args, n).ok_or_else(|| format!("{n} is required"));
    let exe = need("--exe")?;
    let net = need("--net")?;
    let book = need("--book")?;
    let knob_file = need("--knobs")?;
    let out_path = need("--out")?;
    let nodes: Vec<u64> = flag(args, "--nodes")
        .unwrap_or_else(|| "5000,50000".to_string())
        .split(',')
        .map(|x| x.trim().parse::<u64>().map_err(|_| format!("bad --nodes entry '{x}'")))
        .collect::<Result<_, _>>()?;
    let warm_nodes: Option<u64> = flag(args, "--warm-nodes").and_then(|x| x.parse().ok());
    let warm_iters: usize = if warm_nodes.is_some() { flag(args, "--warm-iters").and_then(|x| x.parse().ok()).unwrap_or(100) } else { 0 };
    if nodes.iter().any(|&n| n < 5000) {
        return Err("--nodes entries must be >= 5000 (user 2026-11-01: shallower tuning is wrong)".into());
    }
    let pairs: usize = flag(args, "--pairs").and_then(|x| x.parse().ok()).unwrap_or(50);
    let iters: usize = flag(args, "--iters").and_then(|x| x.parse().ok()).unwrap_or(200);
    let threads: usize = flag(args, "--threads").and_then(|x| x.parse().ok()).unwrap_or(4);
    let book0: usize = flag(args, "--book0").and_then(|x| x.parse().ok()).unwrap_or(60000);
    let mut seed: u64 = flag(args, "--seed").and_then(|x| x.parse().ok()).unwrap_or(1) | 1;
    let resume = args.iter().any(|a| a == "--resume");

    let knobs = load_knobs(&knob_file)?;
    let defaults = engine_defaults(&exe)?;
    let mut theta: Vec<f64> = Vec::new();
    for k in &knobs {
        match defaults.get(&k.name) {
            Some(d) => theta.push(*d),
            None => return Err(format!("knob '{}' is not in `{exe} tune-list`", k.name)),
        }
    }
    let mut k0 = 0usize;
    if resume {
        if let Some((k, th)) = last_theta(&out_path, &knobs) {
            k0 = k;
            theta = th;
            println!("[spsa] resume at iteration {k0}");
        }
    } else if Path::new(&out_path).exists() {
        return Err(format!("{out_path} exists: pass --resume or choose a new --out"));
    }
    let me = std::env::current_exe().map_err(|e| e.to_string())?;
    let a_big = 0.1 * iters as f64;
    let mut log = fs::OpenOptions::new().create(true).append(true).open(&out_path).map_err(|e| e.to_string())?;
    if k0 == 0 {
        let _ = writeln!(
            log,
            "# ti spsa | exe {exe} | net {net} | book {book} | knobs {knob_file} ({}) | nodes {:?} | warm {:?} x {warm_iters} | pairs {pairs} | iters {iters} | seed {seed} | defaults {}",
            knobs.len(),
            nodes,
            warm_nodes,
            tune_arg(&knobs, &theta)
        );
    }
    for _ in 0..k0 {
        rng_next(&mut seed);
    }
    for k in k0..iters {
        let ck: Vec<f64> = knobs.iter().map(|kn| (kn.c / ((k + 1) as f64).powf(0.101)).max(1.0)).collect();
        let ak: Vec<f64> = knobs.iter().map(|kn| kn.a / ((k + 1) as f64 + a_big).powf(0.602)).collect();
        let delta: Vec<f64> = knobs.iter().map(|_| if rng_next(&mut seed) & 1 == 0 { 1.0 } else { -1.0 }).collect();
        let clamp = |i: usize, v: f64| v.clamp(knobs[i].lo, knobs[i].hi).round();
        let plus: Vec<f64> = (0..knobs.len()).map(|i| clamp(i, theta[i] + ck[i] * delta[i])).collect();
        let minus: Vec<f64> = (0..knobs.len()).map(|i| clamp(i, theta[i] - ck[i] * delta[i])).collect();
        let nd = match warm_nodes {
            Some(w) if k < warm_iters => w,
            _ => nodes[(k - warm_iters.min(k)) % nodes.len()],
        };
        let cmd = |th: &[f64]| format!("{exe} serve --tune {} --network {net}", tune_arg(&knobs, th));
        let o = Command::new(&me)
            .args(["sprt", "--candidate", &cmd(&plus), "--baseline", &cmd(&minus)])
            .args(["--book", &book, "--book0", &(book0 + k * pairs).to_string(), "--nodes", &nd.to_string()])
            .args(["--max-games", &(pairs * 2).to_string(), "--threads", &threads.to_string()])
            .args(["--elo0", "-1000", "--elo1", "1000"])
            .output()
            .map_err(|e| format!("sprt child: {e}"))?;
        let text = String::from_utf8_lossy(&o.stdout).to_string();
        let (w, l, d) = parse_wld(&text).ok_or_else(|| {
            format!("iteration {k}: no result line from the sprt child (exit {:?}); stderr: {}", o.status.code(), String::from_utf8_lossy(&o.stderr).lines().last().unwrap_or(""))
        })?;
        let games = (w + l + d).max(1) as f64;
        let score = (w as f64 + 0.5 * d as f64) / games;
        let r = 2.0 * score - 1.0;
        for i in 0..knobs.len() {
            theta[i] = (theta[i] + ak[i] * r * delta[i]).clamp(knobs[i].lo, knobs[i].hi);
        }
        let line = format!(
            "k={k} nodes={nd} w={w} l={l} d={d} score={score:.4} R={r:+.3} plus={} theta_next={}",
            tune_arg(&knobs, &plus),
            tune_arg(&knobs, &theta)
        );
        println!("{line}");
        let _ = writeln!(log, "{line}");
    }
    println!("[spsa] done: {}", tune_arg(&knobs, &theta));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pair_line() {
        let out = "x\n[Pair  50 | 100g |  61.3s] + 52 - 41 =  7 ( 55.5%) | Elo +38.1 +- 50.0 | LLR\n";
        assert_eq!(parse_wld(out), Some((52, 41, 7)));
    }

    #[test]
    fn tune_arg_rounds() {
        let k = vec![Knob { name: "a".into(), lo: 0.0, hi: 9.0, c: 1.0, a: 1.0 }, Knob { name: "b".into(), lo: 0.0, hi: 9.0, c: 1.0, a: 1.0 }];
        assert_eq!(tune_arg(&k, &[1.4, 2.6]), "a=1,b=3");
    }
}
