//! `lab eval-gate` (D1067, user: "measure a net alone with the most basic alpha-beta, so the configuration cannot matter and
//! the better eval wins"): two NNUE nets play each other in ONE process under the same plain search:
//! negamax alpha-beta, no TT, no null move, no LMR / futility / SmallNet / policy; the same fixed move order for both
//! (captures first, clones before jumps). Botzone rules: no passes, the game ends when either side is stuck (the mobile
//! side takes the empties), cap 400 plies (then raw counts).
//! Modes: `--depth D` = pure fixed depth (the minimax value is independent of move order, so ONLY the eval decides);
//! `--nodes N` = iterative deepening until N nodes per move (the last completed depth decides; a partial iteration is
//! used only for root moves searched completely, the previous best is searched first).
//! Openings: book lines (Board::from_str), each played twice with the seats swapped = one pair.
//! Usage: lab eval-gate --a NET_A.tnn --b NET_B.tnn [--depth D | --nodes N] [--book F] [--book0 K] [--pairs P]
//!        [--threads T]
//! Output: per-pair (pentanomial) score, Elo of A over B +- 95%, LOS, avg depth / nodes per move.

use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use titanium::nnue::Nnue;
use titanium::{Board, Move, MoveList};

use crate::util::arg;

const WIN: i32 = 1_000_000;
const MAX_PLIES: u32 = 400;

/// D1113: an evaluator = a net file (.tnn) or a residual model on the champion (hce-train `*_weights.json`)
pub enum Ev {
    Net(Nnue),
    Res(Box<crate::hcetrain::Residual>),
    /// D1137: 3x3 tuple network (lab jat3-train `*_jat3.json`)
    Jat(Box<crate::jat3::Jat3Net>),
    /// D1237: Zataxx's own net (net6.bin or a lab zataxx-train `*_net.bin`), bit-exact port of its nn.rs (extval::Zat,
    /// parity 8/8 with zataxx.exe 'eval', D995); score in Zataxx's own cp units (move choice only needs the order)
    Zat(Box<crate::extval::Zat>),
    /// D1200/D1267: trained `*_levatt.bin` lever-attention head on champion NNUE trunk
    Lev(Box<crate::levatt::LevAttNet>),
}

impl Ev {
    /// unrounded cp from the side to move (D1196, lab residual-pair on any two evaluators)
    pub(crate) fn eval_f(&self, b: &Board) -> f32 {
        match self {
            Ev::Net(n) => n.eval(b, 0) as f32,
            Ev::Res(r) => r.eval_f(b),
            Ev::Jat(j) => j.eval_f(b),
            Ev::Zat(z) => z.eval(b.me(), b.opp()) as f32,
            Ev::Lev(l) => l.eval_f(b),
        }
    }
    pub(crate) fn is_res(&self) -> bool {
        matches!(self, Ev::Res(_))
    }
    pub(crate) fn load(p: &str) -> Ev {
        if p.ends_with("net6.bin") || p.ends_with("_net.bin") {
            Ev::Zat(Box::new(crate::extval::Zat::load(p)))
        } else if p.ends_with("_jat3.json") {
            Ev::Jat(Box::new(crate::jat3::Jat3Net::load(p)))
        } else if p.ends_with("_levatt.bin") {
            Ev::Lev(Box::new(crate::levatt::LevAttNet::load(p)))
        } else if p.ends_with(".json") {
            Ev::Res(Box::new(crate::hcetrain::Residual::load(p)))
        } else {
            Ev::Net(Nnue::load(p).expect("load net"))
        }
    }
    fn eval(&self, b: &Board) -> i32 {
        match self {
            Ev::Net(n) => n.eval(b, 0),
            Ev::Res(r) => r.eval(b),
            Ev::Jat(j) => j.eval(b),
            Ev::Zat(z) => z.eval(b.me(), b.opp()),
            Ev::Lev(l) => l.eval(b),
        }
    }
}

struct Ctx<'a> {
    net: &'a Ev,
    nodes: u64,
    budget: u64,
    abort: bool,
}

pub(crate) fn terminal(b: &Board) -> i32 {
    let (blk, wht) = b.bz_counts();
    let (me, op) = if b.turn == 0 { (blk as i32, wht as i32) } else { (wht as i32, blk as i32) };
    let d = me - op;
    if d > 0 {
        WIN + d
    } else if d < 0 {
        -WIN + d
    } else {
        0
    }
}

/// D1280 (lab sibling-corr): value of `b` for its side to move by the same plain alpha-beta, iterative deepening until
/// `budget` nodes (the last completed depth decides). Does not touch the eval-gate path.
pub(crate) fn search_value(net: &Ev, b: &Board, budget: u64) -> i32 {
    if b.bz_over() {
        return terminal(b);
    }
    let mut c = Ctx { net, nodes: 0, budget, abort: false };
    let mut val = net.eval(b);
    for d in 1..=64u32 {
        let v = ab(&mut c, b, d, -2 * WIN, 2 * WIN);
        if c.abort {
            break;
        }
        val = v;
        if v.abs() >= WIN {
            break;
        }
    }
    val
}

pub(crate) fn ordered(b: &Board) -> Vec<Move> {
    let mut ml = MoveList::new();
    b.legal_moves_dedup_into(&mut ml);
    let mut v: Vec<(i32, Move)> = ml.iter().map(|m| ((b.captures(m.to) as i32) * 2 + m.is_clone() as i32, m)).collect();
    v.sort_by(|x, y| y.0.cmp(&x.0).then((x.1.from, x.1.to).cmp(&(y.1.from, y.1.to))));
    v.into_iter().map(|x| x.1).collect()
}

fn ab(c: &mut Ctx, b: &Board, depth: u32, mut alpha: i32, beta: i32) -> i32 {
    c.nodes += 1;
    if c.budget > 0 && c.nodes >= c.budget {
        c.abort = true;
    }
    if b.bz_over() {
        return terminal(b);
    }
    if depth == 0 {
        return c.net.eval(b);
    }
    let mut best = -2 * WIN;
    for m in ordered(b) {
        let v = -ab(c, &b.make(m), depth - 1, -beta, -alpha);
        if c.abort {
            return best.max(v);
        }
        if v > best {
            best = v;
            if v > alpha {
                alpha = v;
                if alpha >= beta {
                    break;
                }
            }
        }
    }
    best
}

/// Root search: returns (move, completed depth, nodes).
fn choose(net: &Ev, b: &Board, depth: u32, budget: u64) -> (Move, u32, u64) {
    let mut root = ordered(b);
    let mut c = Ctx { net, nodes: 0, budget, abort: false };
    let (mut best_m, mut done) = (root[0], 0u32);
    let max_d = if budget > 0 { 64 } else { depth };
    let start_d = if budget > 0 { 1 } else { depth };
    for d in start_d..=max_d {
        let (mut alpha, mut it_best, mut it_full) = (-2 * WIN, root[0], 0usize);
        for (i, &m) in root.iter().enumerate() {
            let v = -ab(&mut c, &b.make(m), d - 1, -2 * WIN, -alpha);
            if c.abort {
                break;
            }
            it_full = i + 1;
            if v > alpha {
                alpha = v;
                it_best = m;
            }
        }
        if !c.abort {
            best_m = it_best;
            done = d;
        } else if it_full > 0 && d > 1 {
            // partial iteration: the previous best was searched first, so the best of the fully searched moves is at
            // least as informed as the previous iteration's choice
            best_m = it_best;
        }
        if c.abort || (budget == 0 && d == depth) {
            break;
        }
        // previous best first for the next iteration
        if let Some(p) = root.iter().position(|&m| m == best_m) {
            let m = root.remove(p);
            root.insert(0, m);
        }
        if alpha.abs() >= WIN {
            break; // solved
        }
    }
    (best_m, done.max(1), c.nodes)
}

struct Game {
    score_a: f64,
    depth: [u64; 2],
    nodes: [u64; 2],
    moves: [u64; 2],
}

fn play(start: &Board, a_first: bool, nets: [&Ev; 2], depth: u32, budget: u64) -> Game {
    // side index 0 = net A, 1 = net B; A moves first from the book position when a_first
    let mut b = *start;
    let first_turn = b.turn;
    let mut g = Game { score_a: 0.5, depth: [0; 2], nodes: [0; 2], moves: [0; 2] };
    let mut plies = 0u32;
    loop {
        if b.bz_over() || plies >= MAX_PLIES {
            let (blk, wht) = if b.bz_over() { b.bz_counts() } else { b.counts() };
            // A's colour
            let a_col = if a_first { first_turn } else { 1 - first_turn };
            let (a, o) = if a_col == 0 { (blk, wht) } else { (wht, blk) };
            g.score_a = if a > o { 1.0 } else if a < o { 0.0 } else { 0.5 };
            return g;
        }
        let side = if (b.turn == first_turn) == a_first { 0 } else { 1 };
        let (m, d, n) = choose(nets[side], &b, depth, budget);
        g.depth[side] += d as u64;
        g.nodes[side] += n;
        g.moves[side] += 1;
        b = b.make(m);
        plies += 1;
    }
}

fn elo(s: f64) -> f64 {
    let s = s.clamp(1e-6, 1.0 - 1e-6);
    -400.0 * (1.0 / s - 1.0).log10()
}

pub fn run(args: &[String]) -> ExitCode {
    let pa = arg(args, "--a").expect("--a NET_A.tnn");
    let pb = arg(args, "--b").expect("--b NET_B.tnn");
    let depth: u32 = arg(args, "--depth").map_or(0, |v| v.parse().unwrap());
    let budget: u64 = arg(args, "--nodes").map_or(0, |v| v.parse().unwrap());
    if (depth == 0) == (budget == 0) {
        eprintln!("eval-gate: give exactly one of --depth D or --nodes N");
        return ExitCode::from(2);
    }
    let book = arg(args, "--book").unwrap_or_else(|| "C:/gitProjects/titanium-ataxx/data/nnue/book50k_std5k.txt".into());
    let book0: usize = arg(args, "--book0").map_or(0, |v| v.parse().unwrap());
    let pairs: usize = arg(args, "--pairs").map_or(400, |v| v.parse().unwrap());
    let threads: usize = arg(args, "--threads").map_or(6, |v| v.parse().unwrap());
    // D1130: a trained lab model plays only if it was trained by the standard (whole dataset, champion target);
    // a non-standard model needs --experiment NAME, and the result line says EXPERIMENT
    let exp = crate::standard::experiment(args);
    for p in [&pa, &pb] {
        if p.ends_with(".json") {
            let s = std::fs::read_to_string(p).unwrap_or_default();
            if !s.contains("\"standard\":true") && exp.is_none() {
                eprintln!("eval-gate: {p} was not trained by the lab standard (whole dataset, champion target; no \"standard\":true stamp, D1130). Refusing; pass --experiment NAME to play it anyway.");
                return ExitCode::from(2);
            }
        }
    }
    if let Some(n) = &exp {
        println!("eval-gate: EXPERIMENT {n}: at least one model may be non-standard; this is NOT a standard result");
    }
    let na = Ev::load(&pa);
    let nb = Ev::load(&pb);
    // sanity: eval is from the side to move (a side with far more stones must score higher)
    let up = Board::from_occ(0x0000_0000_00FF_FFFF, 0x1, 0, 0);
    let down = Board::from_occ(0x1, 0x0000_0000_00FF_FFFF, 0, 0);
    for (n, name) in [(&na, "A"), (&nb, "B")] {
        if n.eval(&up) <= n.eval(&down) {
            eprintln!("eval-gate: net {name} does not score from the side to move; refusing");
            return ExitCode::from(1);
        }
    }
    let text = std::fs::read_to_string(&book).expect("read --book");
    let starts: Vec<Board> = text.lines().skip(book0).filter_map(|l| Board::from_str(l.trim())).take(pairs).collect();
    let sha = |p: &str| {
        use std::hash::Hasher;
        let bytes = std::fs::read(p).unwrap_or_default();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        h.write(&bytes);
        format!("{:016X} ({} B)", h.finish(), bytes.len())
    };
    println!(
        "eval-gate: A {pa} [{}] vs B {pb} [{}] | plain alpha-beta, {} | Botzone rules, cap {MAX_PLIES} plies | book {book} @{book0}, {} pairs, {threads} threads",
        sha(&pa),
        sha(&pb),
        if depth > 0 { format!("fixed depth {depth}") } else { format!("{budget} nodes/move (iterative deepening)") },
        starts.len()
    );
    // LAWS A14 (user D1307): every finished pair is appended (and flushed) to --results at once, so a stop, a crash or a
    // power cut keeps every finished game; STOP in the results folder or C:/gitProjects/STOP_ALL is checked before every
    // pair (no new pair starts, the summary of the finished pairs is printed); --resume skips the pairs already in the file.
    let Some(results) = arg(args, "--results") else {
        eprintln!("eval-gate: --results FILE.tsv is required (LAWS A14: every finished pair is kept there; STOP in its folder stops the gate)");
        return ExitCode::from(2);
    };
    let run_dir = std::path::Path::new(&results).parent().map(|p| if p.as_os_str().is_empty() { std::path::PathBuf::from(".") } else { p.to_path_buf() }).unwrap_or_else(|| ".".into());
    if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
        eprintln!("eval-gate: stop file {f} exists: remove it to play");
        return ExitCode::from(2);
    }
    let header = format!("# eval-gate A {pa} [{}] B {pb} [{}] depth {depth} nodes {budget} book {book} @{book0} pairs {}", sha(&pa), sha(&pb), starts.len());
    let resume = args.iter().any(|a| a == "--resume");
    let mut prev: Vec<(usize, f64, [u64; 2], [u64; 2], [u64; 2])> = Vec::new();
    if std::path::Path::new(&results).exists() {
        if !resume {
            eprintln!("eval-gate: {results} exists: continue it with --resume, or delete it to start over");
            return ExitCode::from(2);
        }
        let t = std::fs::read_to_string(&results).unwrap_or_default();
        let mut lines = t.lines();
        if lines.next() != Some(header.as_str()) {
            eprintln!("eval-gate --resume: {results} was written by another setup (nets / budget / book / pairs differ)");
            return ExitCode::from(2);
        }
        for l in lines {
            let v: Vec<&str> = l.split('\t').collect();
            if v.len() == 8 {
                let u = |k: usize| v[k].parse::<u64>().unwrap_or(0);
                prev.push((u(0) as usize, v[1].parse().unwrap_or(0.5), [u(2), u(3)], [u(4), u(5)], [u(6), u(7)]));
            }
        }
        println!("eval-gate: RESUMED from {results}: {} finished pairs kept", prev.len());
    } else if std::fs::write(&results, format!("{header}\n")).is_err() {
        eprintln!("eval-gate: cannot write {results}");
        return ExitCode::from(1);
    }
    let done: std::collections::HashSet<usize> = prev.iter().map(|x| x.0).collect();
    let out = Mutex::new(std::fs::OpenOptions::new().append(true).open(&results).expect("open --results"));
    let stopped: Mutex<Option<String>> = Mutex::new(None);
    let next = AtomicUsize::new(0);
    let res: Mutex<Vec<(usize, f64, [u64; 2], [u64; 2], [u64; 2])>> = Mutex::new(prev);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                if let Some(f) = crate::stopctl::requested(Some(&run_dir)) {
                    let mut st = stopped.lock().unwrap();
                    if st.is_none() {
                        println!("eval-gate: STOP requested by {f}: no new pair starts; finished pairs are kept in {results}");
                        *st = Some(f);
                    }
                    break;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= starts.len() {
                    break;
                }
                if done.contains(&i) {
                    continue;
                }
                let g1 = play(&starts[i], true, [&na, &nb], depth, budget);
                let g2 = play(&starts[i], false, [&na, &nb], depth, budget);
                let add = |x: [u64; 2], y: [u64; 2]| [x[0] + y[0], x[1] + y[1]];
                let row = (i, (g1.score_a + g2.score_a) / 2.0, add(g1.depth, g2.depth), add(g1.nodes, g2.nodes), add(g1.moves, g2.moves));
                {
                    use std::io::Write;
                    let mut f = out.lock().unwrap();
                    let _ = writeln!(f, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", row.0, row.1, row.2[0], row.2[1], row.3[0], row.3[1], row.4[0], row.4[1]);
                    let _ = f.flush();
                }
                let mut r = res.lock().unwrap();
                r.push(row);
                let n = r.len();
                if n % 50 == 0 || n == starts.len() {
                    report(&r, n == starts.len());
                }
            });
        }
    });
    let r = res.into_inner().unwrap();
    if let Some(f) = stopped.into_inner().unwrap() {
        if !r.is_empty() {
            report(&r, false);
        }
        println!("eval-gate: STOPPED by {f} after {} of {} pairs (all finished pairs are in {results}); the same command + --resume plays the rest", r.len(), starts.len());
    } else if r.len() == starts.len() && r.len() == done.len() && !r.is_empty() {
        // resumed a run whose pairs were all finished already: print its final line
        report(&r, true);
    }
    ExitCode::SUCCESS
}

/// In-process paired match between two loaded evaluators (used by `lab spsa-blend`).
/// Returns `(mean_score_a, penta)` across `starts` (each played twice with swapped seats).
#[allow(dead_code)]
pub(crate) fn play_match(na: &Ev, nb: &Ev, starts: &[Board], depth: u32, budget: u64, threads: usize) -> (f64, [usize; 5]) {
    let next = AtomicUsize::new(0);
    let res: Mutex<Vec<f64>> = Mutex::new(Vec::with_capacity(starts.len()));
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                // LAWS A14: a stop file ends the match early (the caller sees the stop and discards the partial match)
                if crate::stopctl::requested_global().is_some() {
                    break;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= starts.len() {
                    break;
                }
                let g1 = play(&starts[i], true, [na, nb], depth, budget);
                let g2 = play(&starts[i], false, [na, nb], depth, budget);
                res.lock().unwrap().push((g1.score_a + g2.score_a) / 2.0);
            });
        }
    });
    let r = res.into_inner().unwrap();
    let n = r.len().max(1) as f64;
    let m = r.iter().sum::<f64>() / n;
    let mut penta = [0usize; 5];
    for &x in &r {
        penta[(x * 4.0).round() as usize] += 1;
    }
    (m, penta)
}

fn report(r: &[(usize, f64, [u64; 2], [u64; 2], [u64; 2])], last: bool) {
    let n = r.len() as f64;
    let m = r.iter().map(|x| x.1).sum::<f64>() / n;
    let var = r.iter().map(|x| (x.1 - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let se = (var / n).sqrt();
    let (lo, hi) = (elo(m - 1.96 * se), elo(m + 1.96 * se));
    let z = if se > 0.0 { (m - 0.5) / se } else { 0.0 };
    let los = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    let mut penta = [0usize; 5];
    for x in r {
        penta[(x.1 * 4.0).round() as usize] += 1;
    }
    let sum = |k: usize, f: fn(&(usize, f64, [u64; 2], [u64; 2], [u64; 2])) -> [u64; 2]| r.iter().map(|x| f(x)[k]).sum::<u64>() as f64;
    let (ma, mb) = (sum(0, |x| x.4), sum(1, |x| x.4));
    println!(
        "{}[{:4} pairs] A score {:.1}% | Elo A-B {:+.1} [{:+.1}, {:+.1}] | LOS {:.1}% | penta {:?} | depth A {:.2} B {:.2} | nodes/move A {:.0} B {:.0}",
        if last { "FINAL " } else { "" },
        r.len(),
        100.0 * m,
        elo(m),
        lo,
        hi,
        100.0 * los,
        penta,
        sum(0, |x| x.2) / ma,
        sum(1, |x| x.2) / mb,
        sum(0, |x| x.3) / ma,
        sum(1, |x| x.3) / mb
    );
}

fn erf(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}
