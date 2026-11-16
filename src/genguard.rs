//! LAWS A12 (user order D927, 2026-11-16): ONE idiot-proof generator, no legacy generation.
//!
//! Every command that creates games or labels for the dataset (`ti gen`, `ti relabel`) calls
//! [`require_main_engine`] before it plays or searches anything. The check is BEHAVIOURAL: the engine linked into this
//! `ti` binary must search a fixed set of positions node-for-node like the canonical `main` gate exe
//! (`_gate/exes/titanium-<origin/main>.exe`, built from a clean checkout and verified against BUNDLE.txt by
//! `gate::ensure_main_champion`). Any difference in search, pruning, ordering, big net, policy head or SmallNet changes
//! the node counts, so a stale hub binary, a binary linked against an experiment worktree, or a binary from before a
//! promotion is refused. There is no flag to skip the check.
//!
//! The dataset standard is fixed here as well: production TT size and at least [`STD_NODES`] nodes per move.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use titanium::{Board, SearchLimits, Searcher};

/// nodes per move of the dataset (D787: 5,000-node self-play and labels); fewer is refused, more is allowed
pub const STD_NODES: u64 = 5000;
/// transposition table bits of the played engine (`serve --tt-bits 22`, Botzone package)
pub const PROD_TT_BITS: usize = 22;
/// depth of the identity searches (fast: well under a second per position)
const CHECK_DEPTH: u8 = 7;
/// number of book positions searched for the identity check
const CHECK_POSITIONS: usize = 12;

/// What the check proved; written into every generation run so a file always says which engine made it.
pub struct EngineId {
    pub main_commit: String,
    pub main_exe_sha: String,
    pub main_version: String,
    pub positions: usize,
    pub nodes: u64,
}

impl EngineId {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "rule": "LAWS A12 / D927: games and labels only from the current main engine",
            "main_commit": self.main_commit,
            "main_exe_sha": self.main_exe_sha,
            "main_version": self.main_version,
            "identity_check": format!("{} book positions at depth {CHECK_DEPTH}: node counts and best moves equal to the main gate exe ({} nodes)", self.positions, self.nodes),
            "tt_bits": PROD_TT_BITS,
        })
    }
}

/// Flags that would weaken a generator or labeller below the dataset standard. Refused, never documented as options.
pub fn refuse_weakening_flags(cmd: &str, args: &[String], banned: &[&str]) -> Result<(), String> {
    for b in banned {
        if args.iter().any(|a| a == b || a.starts_with(&format!("{b}="))) {
            return Err(format!(
                "{cmd}: {b} is refused (LAWS A12, D927): games and labels are made only by the current main engine with its own \
                 net, policy head and SmallNet, production TT 2^{PROD_TT_BITS} and >= {STD_NODES} nodes per move"
            ));
        }
    }
    Ok(())
}

/// Refuse a node budget below the dataset standard.
pub fn check_nodes(cmd: &str, nodes: u64) -> Result<(), String> {
    if nodes < STD_NODES {
        return Err(format!("{cmd}: --nodes {nodes} is below the dataset standard {STD_NODES} (LAWS A12, D927)"));
    }
    Ok(())
}

/// The behavioural identity check described in the module comment.
pub fn require_main_engine(cmd: &str) -> Result<EngineId, String> {
    // always compare against the LATEST main on GitHub (offline: the local origin/main, with a warning)
    let f = Command::new("git").args(["fetch", "-q", "origin"]).current_dir("C:/gitProjects/mainpromo-ataxx").status();
    if !f.map(|s| s.success()).unwrap_or(false) {
        eprintln!("{cmd}: warning: git fetch origin failed; checking against the local origin/main");
    }
    let main = crate::gate::ensure_main_champion().map_err(|e| format!("{cmd}: cannot resolve the main engine: {e}"))?;
    let root = crate::gate::engine_root()?;
    let book_path = crate::gate::default_book(&root);
    let text = std::fs::read_to_string(&book_path).map_err(|e| format!("{cmd}: book {}: {e}", book_path.display()))?;
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.len() < CHECK_POSITIONS {
        return Err(format!("{cmd}: book {} too small for the identity check", book_path.display()));
    }
    let step = lines.len() / CHECK_POSITIONS;
    let boards: Vec<Board> = (0..CHECK_POSITIONS)
        .map(|i| Board::from_str(lines[i * step]).ok_or_else(|| format!("{cmd}: bad book line {}", i * step)))
        .collect::<Result<_, _>>()?;

    let mut child = Command::new(&main.exe)
        .args(["serve", "--tt-bits", &PROD_TT_BITS.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("{cmd}: spawn {}: {e}", main.exe.display()))?;
    let mut stdin = child.stdin.take().ok_or("no stdin")?;
    let mut out = BufReader::new(child.stdout.take().ok_or("no stdout")?);
    let mut send = |l: &str| -> Result<(), String> { writeln!(stdin, "{l}").and_then(|_| stdin.flush()).map_err(|e| e.to_string()) };
    let mut read_until = |prefix: &str| -> Result<Vec<String>, String> {
        let mut seen = Vec::new();
        loop {
            let mut s = String::new();
            if out.read_line(&mut s).map_err(|e| e.to_string())? == 0 {
                return Err(format!("main exe closed while waiting for '{prefix}'"));
            }
            let t = s.trim().to_string();
            let done = t.starts_with(prefix);
            seen.push(t);
            if done {
                return Ok(seen);
            }
        }
    };
    send("uai")?;
    read_until("uaiok")?;
    let mut total = 0u64;
    let mut result = Ok(());
    for (i, b) in boards.iter().enumerate() {
        send("uainewgame")?;
        send("isready")?;
        read_until("readyok")?;
        send(&format!("position fen {}", b.to_ataxx_fen()))?;
        send(&format!("go depth {CHECK_DEPTH}"))?;
        let seen = read_until("bestmove")?;
        let main_nodes = seen
            .iter()
            .rev()
            .filter(|l| l.starts_with("info"))
            .find_map(|l| l.split_whitespace().skip_while(|t| *t != "nodes").nth(1).and_then(|n| n.parse::<u64>().ok()));
        let main_best = seen.last().and_then(|l| l.split_whitespace().nth(1)).unwrap_or("").to_string();

        let mut se = Searcher::with_tt_bits(PROD_TT_BITS);
        let r = se.search(b, &SearchLimits { time: None, max_nodes: None, max_depth: CHECK_DEPTH });
        let main_mv = b.parse_uai_move(&main_best);
        let mine_best = r.best.map(|m| format!("{m:?}")).unwrap_or_default();
        total += r.nodes;
        if main_nodes != Some(r.nodes) || main_mv != r.best {
            result = Err(format!(
                "{cmd}: REFUSED (LAWS A12, D927): the engine linked into this ti binary is NOT the current main {} \
                 (position {i}: main exe {:?} nodes / best {main_best}, this binary {} nodes / best {mine_best}). \
                 Rebuild ti against a clean checkout of origin/main (titaniumtest/.cargo/config.toml patch -> mainpromo-ataxx at origin/main).",
                main.commit, main_nodes, r.nodes
            ));
            break;
        }
    }
    let _ = send("quit");
    let _ = child.wait();
    result?;
    Ok(EngineId {
        main_commit: main.commit.clone(),
        main_exe_sha: main.exe_sha.clone(),
        main_version: main.version.clone(),
        positions: CHECK_POSITIONS,
        nodes: total,
    })
}

/// D931 (user): experiments are allowed, but never confusable with the standard path. `--experiment NAME` is the ONLY
/// switch that unlocks overrides; its outputs must live under a folder named `experiments/`.
pub fn experiment(args: &[String]) -> Option<String> {
    crate::flag_str(args, "--experiment").filter(|s| !s.is_empty())
}

/// true when `p` lies under a folder named `experiments` (either slash style)
pub fn is_experimental_path(p: &str) -> bool {
    p.replace('\\', "/").split('/').any(|c| c == "experiments")
}

/// Experiments write only under experiments/; the standard path never writes there and never reads from there.
pub fn check_output(cmd: &str, exp: &Option<String>, out: &str) -> Result<(), String> {
    match (exp, is_experimental_path(out)) {
        (Some(_), true) | (None, false) => Ok(()),
        (Some(n), false) => Err(format!("{cmd}: --experiment {n} writes only under a folder named experiments/ (got {out}) (D931)")),
        (None, true) => Err(format!("{cmd}: {out} is under experiments/: the standard path never writes there; pass --experiment NAME (D931)")),
    }
}

/// The standard path refuses experimental inputs (so a standard dataset cannot silently contain experiment output).
pub fn check_inputs<'a>(cmd: &str, exp: &Option<String>, inputs: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    if exp.is_some() {
        return Ok(());
    }
    for p in inputs {
        if is_experimental_path(p) {
            return Err(format!("{cmd}: input {p} is experimental (under experiments/); the standard path does not read it; pass --experiment NAME (D931)"));
        }
    }
    Ok(())
}

/// Banner every experimental run prints first.
pub fn announce(cmd: &str, exp: &Option<String>) {
    if let Some(n) = exp {
        eprintln!("{cmd}: *** EXPERIMENT '{n}' *** NOT the standard path: overrides allowed, outputs under experiments/ only (D931)");
    }
}

/// User 2026-11-16 ("the hub always pulls the latest main and rebuilds itself if its build is stale before generation
/// or relabelling"): called by the STANDARD path when the identity check fails. Steps: (1) `git fetch origin` in the
/// main checkout and fast-forward it to origin/main (refused if that checkout has tracked changes or cannot ff);
/// (2) rebuild this hub against it (`cargo build` in the hub's own source folder, target_main); (3) re-run the SAME
/// command with the new binary and exit with its code. Never returns on success. A second failure after a rebuild is
/// a hard refusal (env TI_SELF_REBUILT guards the loop).
pub fn rebuild_and_rerun(cmd: &str, why: &str) -> Result<(), String> {
    if std::env::var("TI_SELF_REBUILT").is_ok() {
        return Err(format!("{cmd}: still not the current main after a self-rebuild: {why}"));
    }
    let main_wt = std::path::PathBuf::from("C:/gitProjects/mainpromo-ataxx");
    let git = |args: &[&str]| -> Result<String, String> {
        let o = Command::new("git").args(args).current_dir(&main_wt).output().map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(format!("git {}: {}", args.join(" "), String::from_utf8_lossy(&o.stderr).trim()));
        }
        Ok(String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    eprintln!("{cmd}: this ti binary is not the current main ({why}); updating main and rebuilding ti (user rule 2026-11-16) ...");
    if let Err(e) = git(&["fetch", "-q", "origin"]) {
        eprintln!("{cmd}: warning: git fetch failed ({e}); using the local origin/main");
    }
    // only the engine parts decide what ti links (docs / ledger edits in that checkout do not matter)
    let dirty = git(&["status", "--porcelain", "-uno", "--", "crates", "data/nnue", "Cargo.toml", "Cargo.lock"])?;
    if !dirty.is_empty() {
        return Err(format!("{cmd}: {} has engine changes ({dirty}); it must be a clean origin/main checkout (B0). Refusing.", main_wt.display()));
    }
    if git(&["branch", "--show-current"])? != "main" {
        return Err(format!("{cmd}: {} is not on branch main. Refusing.", main_wt.display()));
    }
    git(&["merge", "--ff-only", "-q", "origin/main"])?;
    let hub_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe_rel = "target_main/x86_64-pc-windows-gnu/release/ti.exe";
    let new_exe = hub_dir.join(exe_rel);
    // Windows: a running exe cannot be overwritten but can be renamed; move it aside if this process IS that file.
    if let Ok(me) = std::env::current_exe() {
        if me.canonicalize().ok() == new_exe.canonicalize().ok() && new_exe.exists() {
            let aside = new_exe.with_extension(format!("exe.old-{}", std::process::id()));
            std::fs::rename(&new_exe, &aside).map_err(|e| format!("{cmd}: cannot move the running ti aside: {e}"))?;
        }
    }
    let path = format!(r"C:\msys64\ucrt64\bin;{}", std::env::var("PATH").unwrap_or_default());
    let st = Command::new("cargo")
        .args(["build", "--release", "--bin", "ti", "--target", "x86_64-pc-windows-gnu", "--target-dir", "target_main"])
        .current_dir(&hub_dir)
        .env("PATH", path)
        .status()
        .map_err(|e| format!("{cmd}: cargo: {e}"))?;
    if !st.success() || !new_exe.exists() {
        return Err(format!("{cmd}: rebuilding ti failed; nothing was generated"));
    }
    eprintln!("{cmd}: rebuilt {} against main {}; re-running the same command", new_exe.display(), git(&["rev-parse", "--short=8", "HEAD"])?);
    let st = Command::new(&new_exe).args(std::env::args().skip(1)).env("TI_SELF_REBUILT", "1").status().map_err(|e| e.to_string())?;
    std::process::exit(st.code().unwrap_or(1));
}
