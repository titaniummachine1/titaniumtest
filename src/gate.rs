//! `ti gate`, `ti log`, `ti merge-check`: compare two commits (or two nets in the same engine) the same way every time,
//! and keep ONE record per run so we always know what passed, what failed, and where to roll back to.
//!
//!   ti gate --test REF [--base main] --kind search|eval|speed|rules|nonfunctional|tooling [--nodes 5000 | --time-ms 100]
//!   ti gate --net-test NEW.tnn [--net-base data/nnue/nnue_main.tnn] [--ref HEAD] --kind smaller|bigger|same|retrain
//!   ti log [--n 20]                       the rollback map (newest last)
//!   ti merge-check REF [--base main]      READY only with a PASS record for exactly these commits (+ signature rule)
//!   ti merge-check --net NEW.tnn          READY only with a PASS net record vs the current main net
//!
//! Each side is built from a clean checkout of its commit (never the working tree), and its `titanium-cli version`
//! must say that commit with dirty=0 (builds from before `version` existed are recorded as unsupported).
//! The verdict in the record is recomputed from `<log>.games.jsonl` (paired pentanomial SPRT), not copied from output.
//! Records live in docs/GATES.jsonl: flat JSON, one line per run, appended, never edited.

use crate::games::{field, num};
use crate::sprt::{SprtTally, SprtVerdict, MIN_GAMES};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use crate::hash;

const EXE: &str = if cfg!(windows) { "titanium-cli.exe" } else { "titanium-cli" };
const REF_KINDS: [&str; 6] = ["search", "eval", "speed", "rules", "nonfunctional", "tooling"];
const NET_KINDS: [&str; 4] = ["smaller", "bigger", "same", "retrain"];
pub const RECORDS: &str = "docs/GATES.jsonl";

// ---------------------------------------------------------------- small helpers

fn sh(prog: &str, args: &[&str], dir: Option<&Path>, env: &[(&str, &Path)]) -> Result<String, String> {
    let mut c = Command::new(prog);
    c.args(args);
    if let Some(d) = dir {
        c.current_dir(d);
    }
    for (k, v) in env {
        c.env(k, v);
    }
    let o = c.output().map_err(|e| format!("{prog}: {e}"))?;
    if o.status.success() {
        Ok(String::from_utf8_lossy(&o.stdout).trim().to_string())
    } else {
        let err = String::from_utf8_lossy(&o.stderr);
        let tail: Vec<&str> = err.lines().rev().take(12).collect();
        Err(format!("{prog} {} failed:\n{}", args.join(" "), tail.into_iter().rev().collect::<Vec<_>>().join("\n")))
    }
}

fn git(dir: &Path, args: &[&str]) -> Result<String, String> {
    sh("git", args, Some(dir), &[])
}

fn fwd(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

fn civil(z0: i64) -> (i64, u32, u32) {
    let z = z0 + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn iso_utc(secs: u64) -> String {
    let (y, m, d) = civil((secs / 86400) as i64);
    let r = secs % 86400;
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", r / 3600, r % 3600 / 60, r % 60)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Timed gates need an idle box: refuse when another engine / gate process is running.
fn busy_processes() -> Vec<String> {
    if !cfg!(windows) {
        return vec![];
    }
    let Ok(out) = Command::new("tasklist").args(["/FO", "CSV", "/NH"]).output() else { return vec![] };
    let mut found = vec![];
    let mut ti_count = 0;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let name = line.split(',').next().unwrap_or("").trim_matches('"').to_lowercase();
        if name == "ti.exe" {
            ti_count += 1;
        } else if ["titanium-cli.exe", "zataxx.exe", "cargo.exe", "rustc.exe", "python.exe"].contains(&name.as_str())
            || name.starts_with("moonbird")
        {
            found.push(name);
        }
    }
    if ti_count > 1 {
        found.push("ti.exe (another)".into());
    }
    found.sort();
    found.dedup();
    found
}

// ---------------------------------------------------------------- building a side

#[derive(Clone, Debug)]
pub struct Side {
    pub label: String,
    pub commit: String,
    pub exe: PathBuf,
    pub exe_sha: String,
    pub net: String,
    pub net_path: Option<PathBuf>,
    pub version: String,
}

impl Side {
    pub fn spec(&self) -> String {
        match &self.net_path {
            Some(n) => format!("{} serve --network {}", fwd(&self.exe), fwd(n)),
            None => format!("{} serve", fwd(&self.exe)),
        }
    }
}

/// Locate the titanium-ataxx engine repository root from any working directory.
pub fn engine_root() -> Result<PathBuf, String> {
    if let Ok(top) = git(Path::new("."), &["rev-parse", "--show-toplevel"]) {
        let p = PathBuf::from(&top);
        if p.join("crates/titanium").exists() {
            return Ok(p);
        }
    }
    for cand in ["C:/gitProjects/mainpromo-ataxx", "C:/gitProjects/titanium-ataxx"] {
        let p = PathBuf::from(cand);
        if p.join("crates/titanium").exists() {
            return Ok(p);
        }
    }
    Err("could not locate titanium-ataxx engine repository".into())
}

/// Locate the canonical balanced 4-corner opening book across all worktrees of the engine repo.
pub fn default_book(root: &Path) -> PathBuf {
    let mut roots = vec![root.to_path_buf()];
    for l in git(root, &["worktree", "list", "--porcelain"]).unwrap_or_default().lines() {
        if let Some(w) = l.strip_prefix("worktree ") {
            roots.push(PathBuf::from(w));
        }
    }
    for name in ["data/nnue/book50k_std5k.txt", "data/nnue/book50k_std.txt"] {
        if let Some(p) = roots.iter().map(|r| r.join(name)).find(|p| p.exists()) {
            return p;
        }
    }
    root.join("data/nnue/book50k_std5k.txt")
}

/// Resolve the canonical `main` commit (comparing `main` and `origin/main` so a stale local ref is never used).
pub fn resolve_main_commit(root: &Path) -> Result<String, String> {
    let local_main = git(root, &["rev-parse", "--verify", "main^{commit}"]).ok();
    let origin_main = git(root, &["rev-parse", "--verify", "origin/main^{commit}"]).ok();
    match (local_main, origin_main) {
        (Some(m), Some(om)) => {
            if m == om {
                Ok(m)
            } else if Command::new("git").args(["merge-base", "--is-ancestor", &m, &om]).current_dir(root).status().map(|s| s.success()).unwrap_or(false) {
                Ok(om)
            } else if Command::new("git").args(["merge-base", "--is-ancestor", &om, &m]).current_dir(root).status().map(|s| s.success()).unwrap_or(false) {
                Ok(m)
            } else {
                Err(format!("local `main` ({}) and `origin/main` ({}) have diverged! Reconcile `main` before running a gate.", &m[..8], &om[..8]))
            }
        }
        (None, Some(om)) => Ok(om),
        (Some(m), None) => Ok(m),
        (None, None) => Err("neither `main` nor `origin/main` exists in git repo".into()),
    }
}

fn parse_version(line: &str) -> Option<(String, String, String)> {
    if !line.starts_with("titanium-cli ") {
        return None;
    }
    let get = |k: &str| line.split_whitespace().find_map(|t| t.strip_prefix(&format!("{k}=")).map(str::to_string));
    Some((get("commit")?, get("dirty")?, get("net")?))
}

/// Parse `titanium-cli info` output (`adb3af4`+ defaults guard).
/// Returns `(bundled_net_sha8, bundled_pol_sha8, bundled_small_sha8, active_is_bundled, policy_on, small_on, has_warning)`.
fn parse_info_bundle(text: &str) -> Option<(String, String, String, bool, bool, bool, bool)> {
    if !text.starts_with("titanium ") || !text.contains("bundled big net") {
        return None;
    }
    let mut net = None;
    let mut pol = None;
    let mut small = None;
    let mut active_bundled = false;
    let mut pol_on = false;
    let mut small_on = false;
    for l in text.lines() {
        let l = l.trim();
        if let Some(r) = l.strip_prefix("bundled big net") {
            net = r.split_whitespace().next().map(str::to_string);
        } else if let Some(r) = l.strip_prefix("bundled policy") {
            pol = r.split_whitespace().next().map(str::to_string);
        } else if let Some(r) = l.strip_prefix("bundled small net") {
            small = r.split_whitespace().next().map(str::to_string);
        } else if let Some(r) = l.strip_prefix("active big net") {
            active_bundled = r.trim() == "bundled";
        } else if let Some(r) = l.strip_prefix("policy head") {
            pol_on = r.trim().starts_with("ON");
        } else if let Some(r) = l.strip_prefix("futility small net") {
            small_on = r.trim().starts_with("ON");
        }
    }
    Some((net?, pol?, small?, active_bundled, pol_on, small_on, text.contains("WARNING:")))
}

/// Find an existing clean worktree checked out at `full_commit`, if any.
fn find_clean_worktree_at(root: &Path, full_commit: &str) -> Option<PathBuf> {
    let list = git(root, &["worktree", "list", "--porcelain"]).ok()?;
    let mut cur_wt: Option<PathBuf> = None;
    let mut cur_head: Option<String> = None;
    for line in list.lines().chain(std::iter::once("")) {
        if let Some(w) = line.strip_prefix("worktree ") {
            cur_wt = Some(PathBuf::from(w));
            cur_head = None;
        } else if let Some(h) = line.strip_prefix("HEAD ") {
            cur_head = Some(h.trim().to_string());
        } else if line.is_empty() {
            if let (Some(wt), Some(h)) = (cur_wt.take(), cur_head.take()) {
                if h == full_commit && wt.exists() {
                    if let Ok(st) = git(&wt, &["status", "--porcelain", "-uno"]) {
                        if st.trim().is_empty() {
                            return Some(wt);
                        }
                    }
                }
            }
        }
    }
    None
}

pub fn ensure_exe(root: &Path, gate_dir: &Path, refname: &str) -> Result<Side, String> {
    let is_main = refname == "main" || refname == "origin/main";
    let full = if is_main {
        resolve_main_commit(root)?
    } else {
        git(root, &["rev-parse", "--verify", &format!("{refname}^{{commit}}")])?
    };
    let short = full[..8].to_string();
    let exes = gate_dir.join("exes");
    fs::create_dir_all(&exes).map_err(|e| e.to_string())?;
    let exe = exes.join(format!("titanium-{short}{}", if cfg!(windows) { ".exe" } else { "" }));
    if !exe.exists() {
        if let Some(clean_wt) = find_clean_worktree_at(root, &full) {
            println!("gate: building {refname} ({short}) from clean worktree {} ...", fwd(&clean_wt));
            sh("cargo", &["build", "--release", "-p", "titanium-cli"], Some(&clean_wt), &[])?;
            fs::copy(clean_wt.join("target").join("release").join(EXE), &exe).map_err(|e| e.to_string())?;
        } else {
            let wt = gate_dir.join(format!("wt-{short}"));
            if wt.exists() {
                let _ = git(root, &["worktree", "remove", "--force", &fwd(&wt)]);
            }
            git(root, &["worktree", "add", "--detach", &fwd(&wt), &full])?;
            let target = gate_dir.join("target");
            println!("gate: building {refname} ({short}) from a clean checkout ...");
            let built = sh("cargo", &["build", "--release", "-p", "titanium-cli"], Some(&wt), &[("CARGO_TARGET_DIR", &target)]);
            let copied = built.and_then(|_| fs::copy(target.join("release").join(EXE), &exe).map(|_| ()).map_err(|e| e.to_string()));
            let _ = git(root, &["worktree", "remove", "--force", &fwd(&wt)]);
            copied?;
        }
    }
    let info_out = Command::new(&exe).arg("info").output().map_err(|e| e.to_string())?;
    let info_text = String::from_utf8_lossy(&info_out.stdout).trim().to_string();
    let (version, net) = if let Some((net_sha, pol_sha, sn_sha, active_bundled, pol_on, sn_on, has_warn)) = parse_info_bundle(&info_text) {
        if is_main {
            if !active_bundled || !pol_on || !sn_on || has_warn {
                return Err(format!(
                    "main champion ({short}) failed defaults guard in `titanium-cli info`:\n{info_text}"
                ));
            }
            if let Ok(bundle_txt) = git(root, &["show", &format!("{full}:data/nnue/BUNDLE.txt")]) {
                if let Some(pin_net) = bundle_txt.lines().find_map(|l| l.strip_prefix("net ")) {
                    let expected_sha8 = pin_net.trim()[..8].to_uppercase();
                    if net_sha != expected_sha8 {
                        return Err(format!(
                            "main binary net mismatch: binary has {net_sha}, BUNDLE.txt at {short} pins {expected_sha8} -> delete {} and retry",
                            fwd(&exe)
                        ));
                    }
                }
            }
        }
        (
            format!("commit={short} dirty=0 net={net_sha} policy={pol_sha} small={sn_sha}"),
            net_sha,
        )
    } else {
        let ver = Command::new(&exe).arg("version").output().map_err(|e| e.to_string())?;
        let line = String::from_utf8_lossy(&ver.stdout).trim().to_string();
        match parse_version(&line) {
            Some((c, dirty, net)) => {
                if c != short || dirty != "0" {
                    return Err(format!("{refname}: binary says commit={c} dirty={dirty}, expected commit={short} dirty=0 -> delete {} and retry", fwd(&exe)));
                }
                (line.clone(), net)
            }
            None => {
                let blob = Command::new("git").args(["show", &format!("{full}:data/nnue/nnue_main.tnn")]).current_dir(root).output().ok();
                let net = blob.filter(|o| o.status.success()).map(|o| hash::short_hex(&o.stdout)).unwrap_or_else(|| "unknown".to_string());
                ("unsupported (build predates `info`/`version`)".to_string(), net)
            }
        }
    };
    Ok(Side { label: refname.to_string(), commit: short, exe_sha: hash::file_label(&fwd(&exe)), exe, net, net_path: None, version })
}

/// Resolve, build (if needed), and verify the canonical `main` champion binary.
pub fn ensure_main_champion() -> Result<Side, String> {
    let root = engine_root()?;
    let gate_dir = root.parent().unwrap_or(&root).join("_gate");
    ensure_exe(&root, &gate_dir, "main")
}

/// One timed run of the fixed signature workload through the side's net: nodes/s, None = unsupported.
fn bench_once(side: &Side) -> Option<u64> {
    let mut c = Command::new(&side.exe);
    c.arg("signature");
    if let Some(n) = &side.net_path {
        c.args(["--network", &fwd(n)]);
    }
    let o = c.output().ok()?;
    String::from_utf8_lossy(&o.stdout).split_whitespace().find_map(|t| t.strip_prefix("nps=")).and_then(|x| x.parse::<u64>().ok())
}

/// Speed of both sides on the fixed workload: INTERLEAVED runs (T,B,T,B,...), median of 5 each, so slow drift of the
/// machine (boost, cache, other load) hits both the same way. 0 = unsupported. A single short run differs by ~15% between
/// identical builds (measured 2026-10-30), so never compare two runs taken at different times.
fn bench_pair(test: &Side, base: &Side) -> (u64, u64) {
    let (mut t, mut b) = (vec![], vec![]);
    for _ in 0..5 {
        match (bench_once(test), bench_once(base)) {
            (Some(x), Some(y)) => {
                t.push(x);
                b.push(y);
            }
            (Some(x), None) => t.push(x),
            (None, Some(y)) => b.push(y),
            (None, None) => {}
        }
    }
    let med = |mut v: Vec<u64>| if v.is_empty() { 0 } else { v.sort(); v[v.len() / 2] };
    (med(t), med(b))
}

#[allow(dead_code)]
/// Legacy single-side bench (kept for tests of the parser path).
fn bench_nps(side: &Side) -> u64 {
    let mut v = vec![];
    for _ in 0..3 {
        let mut c = Command::new(&side.exe);
        c.arg("signature");
        if let Some(n) = &side.net_path {
            c.args(["--network", &fwd(n)]);
        }
        let Ok(o) = c.output() else { return 0 };
        let text = String::from_utf8_lossy(&o.stdout).to_string();
        match text.split_whitespace().find_map(|t| t.strip_prefix("nps=")).and_then(|x| x.parse::<u64>().ok()) {
            Some(n) => v.push(n),
            None => return 0,
        }
    }
    v.sort();
    v[1]
}

/// Nodes per second of one side over the moves actually played in the gate (`info ... nodes N` / wall ms); 0 = no data.
fn game_nps(games: &[crate::games::Game], mover: &str) -> u64 {
    let (mut nodes, mut ms) = (0u64, 0u64);
    for g in games {
        for m in g.moves.iter().filter(|m| m.mover == mover) {
            if let (Some(n), Some(t)) = (m.nodes, m.ms) {
                if t > 0 {
                    nodes += n;
                    ms += t;
                }
            }
        }
    }
    if ms == 0 { 0 } else { nodes * 1000 / ms }
}

/// Headline speed change of a record: games actually played if both sides have it, else the fixed workload.
fn speed_of(r: &Rec) -> String {
    if r.game_test > 0 && r.game_base > 0 {
        format!("{} (games)", pct(r.game_test, r.game_base))
    } else if r.bench_test > 0 && r.bench_base > 0 {
        format!("{} (bench)", pct(r.bench_test, r.bench_base))
    } else {
        "n/a".to_string()
    }
}

fn pct(new: u64, old: u64) -> String {
    if new == 0 || old == 0 { "n/a".into() } else { format!("{:+.1}%", 100.0 * (new as f64 / old as f64 - 1.0)) }
}

// ---------------------------------------------------------------- records

#[derive(Debug, Clone, PartialEq)]
pub struct Rec {
    pub id: String,
    pub mode: String,
    pub kind: String,
    pub test_commit: String,
    pub base_commit: String,
    pub test_net: String,
    pub base_net: String,
    pub games: u32,
    pub verdict: String,
    pub elo: f64,
    pub date: String,
    pub limit: String,
    /// nodes/s of the fixed signature workload (0 = build predates `signature`)
    pub bench_test: u64,
    pub bench_base: u64,
    /// pairs whose two games had the same move list (0/0 = field absent in old records)
    pub identical: u32,
    pub pairs: u32,
    /// nodes/s over the games actually played (both sides ran at the same time: the trustworthy A/B)
    pub game_test: u64,
    pub game_base: u64,
}

pub fn parse_recs(text: &str) -> Vec<Rec> {
    text.lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .filter_map(|l| {
            Some(Rec {
                id: field(l, "id")?.to_string(),
                mode: field(l, "mode")?.to_string(),
                kind: field(l, "kind")?.to_string(),
                test_commit: field(l, "test_commit")?.to_string(),
                base_commit: field(l, "base_commit")?.to_string(),
                test_net: field(l, "test_net")?.to_string(),
                base_net: field(l, "base_net")?.to_string(),
                games: num(l, "games")?,
                verdict: field(l, "verdict")?.to_string(),
                elo: num(l, "elo").unwrap_or(0.0),
                date: field(l, "date")?.to_string(),
                limit: field(l, "limit")?.to_string(),
                bench_test: num(l, "bench_nps_test").unwrap_or(0),
                bench_base: num(l, "bench_nps_base").unwrap_or(0),
                identical: num(l, "identical_pairs").unwrap_or(0),
                pairs: num(l, "pairs").unwrap_or(0),
                game_test: num(l, "game_nps_test").unwrap_or(0),
                game_base: num(l, "game_nps_base").unwrap_or(0),
            })
        })
        .collect()
}

/// The merge rule: a PASS record with >= MIN_GAMES for exactly these commits (ref mode) or these nets (net mode).
pub fn find_pass<'a>(recs: &'a [Rec], mode: &str, test: &str, base: &str) -> Result<&'a Rec, String> {
    let same = |r: &&Rec| {
        r.mode == mode
            && if mode == "net" { r.test_net == test && r.base_net == base } else { r.test_commit == test && r.base_commit == base }
    };
    let mut cand: Vec<&Rec> = recs.iter().filter(same).collect();
    cand.sort_by(|a, b| a.date.cmp(&b.date));
    let Some(last) = cand.last() else {
        return Err(format!("no gate record for {mode} test={test} base={base} (run `ti gate`; a record for an older base does not count)"));
    };
    if last.games < MIN_GAMES {
        return Err(format!("latest record {} has only {} games (< {MIN_GAMES})", last.id, last.games));
    }
    if last.verdict != "PASS" {
        return Err(format!("latest record {} verdict is {} (needs PASS)", last.id, last.verdict));
    }
    Ok(last)
}

fn expected_line(text: &str) -> Option<String> {
    text.lines().find(|l| l.contains("pub const EXPECTED")).map(|l| l.trim().to_string())
}

// ---------------------------------------------------------------- ti gate

fn arg(args: &[String], name: &str) -> Option<String> {
    crate::flag_str(args, name)
}

pub fn run_gate(args: &[String]) -> ExitCode {
    match gate(args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("gate: {e}");
            ExitCode::FAILURE
        }
    }
}

fn gate(args: &[String]) -> Result<ExitCode, String> {
    let root = engine_root()?;
    let gate_dir = root.parent().unwrap_or(&root).join("_gate");
    let net_mode = arg(args, "--net-test").is_some();
    let kind = arg(args, "--kind").ok_or("--kind is required (refs: search|eval|speed|rules|nonfunctional|tooling; nets: smaller|bigger|same|retrain)")?;
    let allowed: &[&str] = if net_mode { &NET_KINDS } else { &REF_KINDS };
    if !allowed.contains(&kind.as_str()) {
        return Err(format!("--kind {kind} not valid here; use one of {}", allowed.join("|")));
    }
    // D552: an unknown flag (e.g. --ms 100) used to be ignored silently and the gate ran at the 5k-node default
    const VALUE_FLAGS: [&str; 14] =
        ["--net-test", "--net-base", "--kind", "--nodes", "--time-ms", "--book", "--ref", "--test", "--base", "--test-args", "--book0", "--elo1", "--max-games", "--threads"];
    const BOOL_FLAGS: [&str; 1] = ["--allow-busy"];
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if VALUE_FLAGS.contains(&a) || a == "--rules" {
            i += 2;
            continue;
        }
        if a.starts_with("--") && !BOOL_FLAGS.contains(&a) {
            return Err(format!("unknown flag {a} (value flags: {}, --rules; switches: --allow-busy)", VALUE_FLAGS.join(" ")));
        }
        i += 1;
    }
    let (nodes, time_ms) = (crate::flag_u64(args, "--nodes"), crate::flag_u64(args, "--time-ms"));
    let limit = match (nodes, time_ms) {
        (Some(_), Some(_)) => return Err("use either --nodes or --time-ms".into()),
        (None, Some(ms)) => format!("ms:{ms}"),
        (n, None) => format!("nodes:{}", n.unwrap_or(5000)),
    };
    if time_ms.is_some() && !args.iter().any(|a| a == "--allow-busy") {
        let busy = busy_processes();
        if !busy.is_empty() {
            return Err(format!("timed gate needs an idle machine, still running: {} (use --allow-busy to override)", busy.join(", ")));
        }
    }
    // default book: the balanced 5k book, looked up in this worktree and every other worktree of the repo (data/ is git-ignored)
    let book = arg(args, "--book").map(PathBuf::from).unwrap_or_else(|| default_book(&root));
    if !book.exists() {
        return Err(format!("book not found: {}", fwd(&book)));
    }

    // ---- sides
    let (mode, test, base) = if net_mode {
        let refname = arg(args, "--ref").unwrap_or_else(|| "HEAD".into());
        let mut t = ensure_exe(&root, &gate_dir, &refname)?;
        let mut b = t.clone();
        let nt = PathBuf::from(arg(args, "--net-test").unwrap());
        let nb = arg(args, "--net-base").map(PathBuf::from).unwrap_or_else(|| root.join("data/nnue/nnue_main.tnn"));
        for n in [&nt, &nb] {
            if !n.exists() {
                return Err(format!("net not found: {}", fwd(n)));
            }
        }
        t.net = hash::file_label(&fwd(&nt));
        b.net = hash::file_label(&fwd(&nb));
        if t.net == b.net {
            return Err("both nets are the same file content (same hash): nothing to gate".into());
        }
        t.net_path = Some(nt);
        b.net_path = Some(nb);
        t.label = format!("{refname}+net {}", t.net);
        b.label = format!("{refname}+net {}", b.net);
        ("net", t, b)
    } else {
        let test_ref = arg(args, "--test").ok_or("--test REF is required (or --net-test FILE)")?;
        let base_ref = arg(args, "--base").unwrap_or_else(|| "main".into());
        let t = ensure_exe(&root, &gate_dir, &test_ref)?;
        let b = ensure_exe(&root, &gate_dir, &base_ref)?;
        // D522: same commit is a valid A/B when the candidate differs only by --test-args (e.g. a --policy file);
        // the args are hashed into the run id and the log header (B14)
        if t.commit == b.commit && arg(args, "--test-args").is_none() {
            return Err("test and base are the same commit (pass --test-args for a file/flag A/B)".into());
        }
        // D416 (user: "local git is fine but it needs a way to track what was gated against what"): the test side is always
        // a commit (ensure_exe); after the run a local annotated tag gate/<run_id> on it records base, limit, args, log.
        ("ref", t, b)
    };

    // ---- run the paired SPRT through the existing `ti sprt` path
    let args_tag = arg(args, "--test-args").map(|a| format!("-a{}", hash::short_hex(a.as_bytes()))).unwrap_or_default();
    // D502: an extension on fresh book lines (--book0 N) is a different run, not a repeat
    let b0_tag = arg(args, "--book0").filter(|v| v != "0").map(|v| format!("-b{v}")).unwrap_or_default();
    let run_id = format!("{}-{}{args_tag}-vs-{}-{}{b0_tag}", &iso_utc(now_secs())[..10], test.commit, base.commit, limit.replace(':', ""));
    let out_dir = root.join("logs").join("gates").join(&run_id);
    fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let log = out_dir.join("sprt.log");
    if log.exists() {
        return Err(format!("{} already exists: this exact gate was already run today", fwd(&log)));
    }
    let elo1 = arg(args, "--elo1").unwrap_or_else(|| "20".into());
    // --test-args "..." = extra engine flags for the test side only (e.g. "--tstg C:/.../tstg_<sha8>.ttst"); the files in it
    // are hashed into the sprt header by repro.rs
    let test_spec = match arg(args, "--test-args") {
        Some(extra) => format!("{} {extra}", test.spec()),
        None => test.spec(),
    };
    let mut sprt: Vec<String> = vec![
        "--candidate".into(), test_spec, "--baseline".into(), base.spec(),
        "--book".into(), fwd(&book),
        "--max-games".into(), arg(args, "--max-games").unwrap_or_else(|| "600".into()),
        "--elo0".into(), "0".into(), "--elo1".into(), elo1.clone(),
        "--out".into(), fwd(&log),
    ];
    match limit.split_once(':') {
        Some(("ms", v)) => sprt.extend(["--time-ms".into(), v.to_string()]),
        Some((_, v)) => sprt.extend(["--nodes".into(), v.to_string()]),
        None => {}
    }
    for flag in ["--threads", "--book0", "--rules"] {
        if let Some(v) = arg(args, flag) {
            sprt.extend([flag.to_string(), v]);
        }
    }
    let (bench_t, bench_b) = bench_pair(&test, &base);
    println!("gate {run_id}\n  test: {} [{}] net {}\n  base: {} [{}] net {}\n  limit {limit}, book {} [{}], kind {kind}\n  speed (fixed workload): test {} nps, base {} nps, change {}", test.label, test.commit, test.net, base.label, base.commit, base.net, fwd(&book), hash::file_label(&fwd(&book)), bench_t, bench_b, pct(bench_t, bench_b));
    crate::run_sprt_cli(&sprt);

    // ---- verdict recomputed from the game records, not from printed text
    let games_path = format!("{}.games.jsonl", fwd(&log));
    let games = crate::games::read_games(&[games_path])?;
    let mut pairs: std::collections::BTreeMap<u64, Vec<f64>> = std::collections::BTreeMap::new();
    for g in &games {
        pairs.entry(g.pair).or_default().push(g.cand_points);
    }
    // identical play: both games of a pair have the same move list (engines that play the same swap colours into the same game)
    let mut by_pair: std::collections::BTreeMap<u64, Vec<Vec<&str>>> = std::collections::BTreeMap::new();
    for g in &games {
        by_pair.entry(g.pair).or_default().push(g.moves.iter().map(|m| m.mv.as_str()).collect());
    }
    let identical = by_pair.values().filter(|v| v.len() == 2 && v[0] == v[1]).count();
    let n_pairs = by_pair.values().filter(|v| v.len() == 2).count();
    let mut tally = SprtTally::default();
    for p in pairs.values().filter(|p| p.len() == 2) {
        tally.record_pair((p[0] * 2.0).round() as u8, (p[1] * 2.0).round() as u8);
    }
    let elo1f: f64 = elo1.parse().unwrap_or(20.0);
    let st = tally.compute(0.0, elo1f, 0.05, 0.05);
    let verdict = if st.games < MIN_GAMES {
        "NO_VERDICT"
    } else {
        match st.verdict {
            SprtVerdict::Pass => "PASS",
            SprtVerdict::Fail => "H0",
            SprtVerdict::Continue => "CONTINUE",
        }
    };
    let rec = format!(
        "{{\"id\":{id},\"date\":{date},\"mode\":\"{mode}\",\"kind\":{kind},\"test_ref\":{tr},\"test_commit\":\"{tc}\",\"test_net\":\"{tn}\",\"test_exe\":\"{te}\",\"base_ref\":{br},\"base_commit\":\"{bc}\",\"base_net\":\"{bn}\",\"base_exe\":\"{be}\",\"book\":\"{bk}\",\"limit\":\"{limit}\",\"bench_nps_test\":{bt},\"bench_nps_base\":{bb},\"game_nps_test\":{gt},\"game_nps_base\":{gb},\"games\":{g},\"w\":{w},\"l\":{l},\"d\":{d},\"penta\":[{p0},{p1},{p2},{p3},{p4}],\"elo\":{elo:.1},\"ci95\":{ci:.1},\"llr\":{llr:.2},\"elo1\":{elo1f},\"identical_pairs\":{ip},\"pairs\":{np},\"seat_split\":{ss:.2},\"verdict\":\"{verdict}\",\"hub\":\"{hub}\",\"log\":{logp}}}",
        id = crate::json_str(&run_id), date = crate::json_str(&iso_utc(now_secs())), kind = crate::json_str(&kind),
        tr = crate::json_str(&test.label), tc = test.commit, tn = test.net, te = test.exe_sha,
        br = crate::json_str(&base.label), bc = base.commit, bn = base.net, be = base.exe_sha,
        bk = hash::file_label(&fwd(&book)), bt = bench_t, bb = bench_b, gt = game_nps(&games, "candidate"), gb = game_nps(&games, "baseline"), g = st.games, w = tally.wins, l = tally.losses, d = tally.draws,
        p0 = tally.penta[0], p1 = tally.penta[1], p2 = tally.penta[2], p3 = tally.penta[3], p4 = tally.penta[4],
        elo = st.elo, ci = st.elo_ci95, llr = st.llr, ip = identical, np = n_pairs, ss = st.seat_split, hub = env!("CARGO_PKG_VERSION"),
        logp = crate::json_str(&format!("logs/gates/{run_id}/sprt.log")),
    );
    let rec_path = root.join(RECORDS);
    if let Some(p) = rec_path.parent() {
        let _ = fs::create_dir_all(p);
    }
    let mut old = fs::read_to_string(&rec_path).unwrap_or_default();
    if !old.is_empty() && !old.ends_with('\n') {
        old.push('\n');
    }
    // D416: test-side engine args and the hash of every file they name (e.g. the .ttst model), so the record alone replays the run
    let test_args = arg(args, "--test-args").unwrap_or_default();
    let test_files = crate::repro::file_args_label(&format!("x {test_args}"), Some(&root));
    let rec = format!(
        "{},\"test_args\":{},\"test_files\":{}}}",
        rec.trim_end_matches('}'),
        crate::json_str(&test_args),
        crate::json_str(test_files.trim())
    );
    fs::write(&rec_path, format!("{old}{rec}\n")).map_err(|e| e.to_string())?;
    // local annotated tag on the tested commit: `git tag -l "gate/*" -n9` lists what was gated against what
    let tag = format!("gate/{run_id}");
    let msg = format!(
        "{verdict} {} g +{} -{} ={} Elo {:+.1} +- {:.1}\ntest {} [{}] args `{test_args}`{test_files}\nbase {} [{}] net {}\nlimit {limit} book {}\nlog logs/gates/{run_id}/sprt.log",
        st.games, tally.wins, tally.losses, tally.draws, st.elo, st.elo_ci95, test.label, test.commit, base.label, base.commit, base.net, hash::file_label(&fwd(&book))
    );
    match git(&root, &["tag", "-a", &tag, &test.commit, "-m", &msg]) {
        Ok(_) => println!("  tagged {tag} on {}", test.commit),
        Err(e) => eprintln!("gate: WARNING could not create tag {tag}: {e}"),
    }
    println!(
        "\nGATE RESULT {verdict}: {} games {}-{}-{}, Elo {:+.1} +- {:.1}, LLR {:+.2} [{:.2}, {:.2}], seat split {:.0}%, identical play in {}/{} pairs  -> recorded in {RECORDS}\n  SPEED test vs base: played games {} ({} vs {} nps) [trust this], fixed workload {} ({} vs {} nps) [noisy]",
        st.games, tally.wins, tally.losses, tally.draws, st.elo, st.elo_ci95, st.llr, st.lower_bound, st.upper_bound, 100.0 * st.seat_split, identical, n_pairs, pct(game_nps(&games, "candidate"), game_nps(&games, "baseline")), game_nps(&games, "candidate"), game_nps(&games, "baseline"), pct(bench_t, bench_b), bench_t, bench_b
    );
    Ok(if verdict == "PASS" { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

// ---------------------------------------------------------------- ti log / ti merge-check

pub fn run_log(args: &[String]) -> ExitCode {
    let root = match engine_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("log: {e}");
            return ExitCode::FAILURE;
        }
    };
    let recs = parse_recs(&fs::read_to_string(root.join(RECORDS)).unwrap_or_default());
    let n = crate::flag_u64(args, "--n").unwrap_or(20) as usize;
    println!("{:<20} {:<13} {:<5} {:<19} {:<10} {:>5} {:>7} {:>14} {:<10} id", "date", "kind", "mode", "test <- base", "limit", "games", "Elo", "speed", "verdict");
    for r in recs.iter().rev().take(n).collect::<Vec<_>>().into_iter().rev() {
        let (t, b) = if r.mode == "net" { (&r.test_net, &r.base_net) } else { (&r.test_commit, &r.base_commit) };
        println!("{:<20} {:<13} {:<5} {:<19} {:<10} {:>5} {:>+7.1} {:>14} {:<10} {}", r.date, r.kind, r.mode, format!("{t} <- {b}"), r.limit, r.games, r.elo, speed_of(r), r.verdict, r.id);
    }
    println!("{} records in {RECORDS}", recs.len());
    ExitCode::SUCCESS
}

pub fn run_merge_check(args: &[String]) -> ExitCode {
    match merge_check(args) {
        Ok(msg) => {
            println!("READY: {msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            println!("NOT READY: {e}");
            ExitCode::FAILURE
        }
    }
}

fn merge_check(args: &[String]) -> Result<String, String> {
    let root = engine_root()?;
    let recs = parse_recs(&fs::read_to_string(root.join(RECORDS)).unwrap_or_default());
    if let Some(net) = arg(args, "--net") {
        let new = hash::file_label(&net);
        let cur = hash::file_label(&fwd(&root.join("data/nnue/nnue_main.tnn")));
        let r = find_pass(&recs, "net", &new, &cur)?;
        return Ok(format!("net {new} beat main net {cur}: record {} ({}, {} games, {:+.1} Elo, speed {})", r.id, r.kind, r.games, r.elo, speed_of(r)));
    }
    let nonfunctional = args.iter().any(|a| a == "--nonfunctional");
    let refname = args.iter().find(|a| !a.starts_with("--") && Some(a.as_str()) != arg(args, "--base").as_deref()).ok_or("usage: ti merge-check REF [--base main] | --net FILE")?;
    let base = arg(args, "--base").unwrap_or_else(|| "main".into());
    let tc = git(&root, &["rev-parse", "--verify", &format!("{refname}^{{commit}}")])?[..8].to_string();
    let bc = if base == "main" || base == "origin/main" {
        resolve_main_commit(&root)?[..8].to_string()
    } else {
        git(&root, &["rev-parse", "--verify", &format!("{base}^{{commit}}")])?[..8].to_string()
    };
    if nonfunctional {
        return nonfunctional_ready(&root, &recs, &tc, &bc);
    }
    let r = find_pass(&recs, "ref", &tc, &bc)?;
    // signature rule: non-functional changes leave EXPECTED untouched; behaviour changes must say so in the commit message
    let sig = "crates/titanium-cli/src/signature.rs";
    let at = |c: &str| git(&root, &["show", &format!("{c}:{sig}")]).ok().and_then(|t| expected_line(&t));
    let (st, sb) = (at(&tc), at(&bc));
    let mut note = String::from("signature n/a");
    if let (Some(a), Some(b)) = (&st, &sb) {
        if a != b {
            if r.kind == "nonfunctional" {
                return Err(format!("kind is nonfunctional but the behaviour signature changed ({b} -> {a})"));
            }
            let msgs = git(&root, &["log", "--format=%B", &format!("{bc}..{tc}")])?;
            if !msgs.contains("Signature:") {
                return Err("behaviour signature changed but no commit message in the range says `Signature: <hex>`".into());
            }
            note = "signature changed and declared".into();
        } else {
            note = "signature unchanged".into();
        }
    }
    Ok(format!("{tc} beat {bc}: record {} ({}, {} games, {:+.1} Elo, speed {}), {note}", r.id, r.kind, r.games, r.elo, speed_of(r)))
}

/// A non-functional change cannot "pass" an Elo SPRT; it must be PROVEN not to change behaviour:
/// the committed signature is untouched, and any gate record for these two commits shows identical play.
fn nonfunctional_ready(root: &Path, recs: &[Rec], tc: &str, bc: &str) -> Result<String, String> {
    let sig = "crates/titanium-cli/src/signature.rs";
    let at = |c: &str| git(root, &["show", &format!("{c}:{sig}")]).ok().and_then(|t| expected_line(&t));
    let (st, sb) = (at(tc), at(bc));
    let sig_note = match (&st, &sb) {
        (Some(a), Some(b)) if a != b => return Err(format!("behaviour signature changed ({b} -> {a}): this is not a non-functional change")),
        (Some(_), Some(_)) => "signature unchanged",
        _ => "signature n/a (a side predates it)",
    };
    let latest = recs.iter().filter(|r| r.mode == "ref" && r.test_commit == tc && r.base_commit == bc).max_by(|a, b| a.date.cmp(&b.date));
    let gate_note = match latest {
        Some(r) if r.pairs > 0 && r.identical == r.pairs => format!("gate record {}: identical play in {}/{} pairs", r.id, r.identical, r.pairs),
        Some(r) if r.pairs > 0 => return Err(format!("gate record {} shows different play in {} of {} pairs: behaviour is NOT unchanged", r.id, r.pairs - r.identical, r.pairs)),
        Some(r) => format!("gate record {} (predates the identical-play field)", r.id),
        None => "no gate record (signature + CI tests are the evidence)".to_string(),
    };
    Ok(format!("{tc} is non-functional vs {bc}: {sig_note}; {gate_note}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: &str, mode: &str, kind: &str, t: &str, b: &str, games: u32, verdict: &str, date: &str) -> String {
        format!("{{\"id\":\"{id}\",\"date\":\"{date}\",\"mode\":\"{mode}\",\"kind\":\"{kind}\",\"test_commit\":\"{t}\",\"test_net\":\"{t}\",\"base_commit\":\"{b}\",\"base_net\":\"{b}\",\"limit\":\"nodes:5000\",\"bench_nps_test\":120000,\"bench_nps_base\":100000,\"games\":{games},\"elo\":12.5,\"verdict\":\"{verdict}\"}}")
    }

    #[test]
    fn iso_date_known_values() {
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_utc(1_793_368_218), "2026-10-30T13:50:18Z");
        assert_eq!(iso_utc(951_782_400), "2000-02-29T00:00:00Z"); // leap day
    }

    #[test]
    fn merge_rule_needs_exact_commits_enough_games_and_pass() {
        let text = [
            rec("a", "ref", "search", "aaaa1111", "bbbb2222", 208, "PASS", "2026-10-30T10:00:00Z"),
            rec("b", "ref", "search", "cccc3333", "bbbb2222", 60, "PASS", "2026-10-30T11:00:00Z"),
            rec("c", "ref", "eval", "dddd4444", "bbbb2222", 400, "H0", "2026-10-30T12:00:00Z"),
        ]
        .join("\n");
        let recs = parse_recs(&text);
        assert_eq!(recs.len(), 3);
        assert!(find_pass(&recs, "ref", "aaaa1111", "bbbb2222").is_ok());
        assert!(find_pass(&recs, "ref", "aaaa1111", "eeee5555").unwrap_err().contains("older base"), "a moved base invalidates the record");
        assert!(find_pass(&recs, "ref", "cccc3333", "bbbb2222").unwrap_err().contains("only 60 games"));
        assert!(find_pass(&recs, "ref", "dddd4444", "bbbb2222").unwrap_err().contains("H0"));
        assert!(find_pass(&recs, "net", "aaaa1111", "bbbb2222").is_err(), "mode must match");
    }

    #[test]
    fn identical_play_fields_roundtrip() {
        let l = "{\"id\":\"x\",\"date\":\"2026-10-30T10:00:00Z\",\"mode\":\"ref\",\"kind\":\"nonfunctional\",\"test_commit\":\"aaaa1111\",\"test_net\":\"N\",\"base_commit\":\"bbbb2222\",\"base_net\":\"N\",\"limit\":\"nodes:2000\",\"games\":100,\"identical_pairs\":50,\"pairs\":50,\"elo\":0.0,\"verdict\":\"CONTINUE\"}";
        let r = &parse_recs(l)[0];
        assert_eq!((r.identical, r.pairs), (50, 50));
    }

    #[test]
    fn speed_is_recorded_and_formatted() {
        let r = &parse_recs(&rec("a", "ref", "search", "aaaa1111", "bbbb2222", 208, "PASS", "2026-10-30T10:00:00Z"))[0];
        assert_eq!((r.bench_test, r.bench_base), (120000, 100000));
        assert_eq!(pct(r.bench_test, r.bench_base), "+20.0%");
        assert_eq!(pct(0, 100), "n/a");
    }

    #[test]
    fn headline_speed_prefers_played_games() {
        let mut r = parse_recs(&rec("a", "ref", "search", "aaaa1111", "bbbb2222", 208, "PASS", "2026-10-30T10:00:00Z")).remove(0);
        assert_eq!(speed_of(&r), "+20.0% (bench)");
        r.game_test = 1000;
        r.game_base = 1001;
        assert_eq!(speed_of(&r), "-0.1% (games)");
    }

    #[test]
    fn latest_record_wins() {
        let text = [
            rec("old", "ref", "search", "aaaa1111", "bbbb2222", 300, "PASS", "2026-10-29T10:00:00Z"),
            rec("new", "ref", "search", "aaaa1111", "bbbb2222", 300, "H0", "2026-10-30T10:00:00Z"),
        ]
        .join("\n");
        assert!(find_pass(&parse_recs(&text), "ref", "aaaa1111", "bbbb2222").unwrap_err().contains("H0"));
    }

    #[test]
    fn version_line_parses() {
        let v = parse_version("titanium-cli commit=5eb4ed00 dirty=0 net=7AE0122F smallnet=8F479ECB arch=x86_64 avx2=1").unwrap();
        assert_eq!(v, ("5eb4ed00".into(), "0".into(), "7AE0122F".into()));
        assert!(parse_version("USAGE: titanium-cli ...").is_none());
    }

    #[test]
    fn signature_line_is_extracted() {
        let t = "x\npub const EXPECTED: u64 = 0xcc8d_cb4e_c15d_2077;\ny";
        assert_eq!(expected_line(t).unwrap(), "pub const EXPECTED: u64 = 0xcc8d_cb4e_c15d_2077;");
    }

    #[test]
    fn info_bundle_parses() {
        let sample = "titanium 0.1.0 | compiled with avx2 true avx512f false\n\
                      bundled big net    46FCB5A4 (29067680 B)\n\
                      bundled policy     AD686DBA (201288 B)\n\
                      bundled small net  8F479ECB\n\
                      active big net     bundled\n\
                      policy head        ON weight 100000 min depth 1\n\
                      futility small net ON\n\
                      build knobs        FUT_CHEAP_M 350 FUT_SN_M 0 EVORD_W 0 EVORD_MIN_DEPTH 1 EVORD_CACHE 1\n";
        let p = parse_info_bundle(sample).unwrap();
        assert_eq!(p, ("46FCB5A4".into(), "AD686DBA".into(), "8F479ECB".into(), true, true, true, false));
    }
}
