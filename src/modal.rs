//! H9: `ti train modal ...` — the only way training jobs reach Modal. Thin: it shells out to `python -m modal` and to the
//! single launcher `modal/ti_modal.py`, applies the cost rules, and keeps an audit row per launch in docs/MODAL_RUNS.jsonl.
//! No credentials are read or stored here: the Modal CLI uses the token in the user's home (~/.modal.toml).
//!
//!   ti train modal put    --files a.tdat,b.manifest.json,... [--dest data] [--force]   upload ONCE to Volume ti-data
//!   ti train modal launch --tag T --args "train run --backend cuda ..." [--timeout-min 30] [--max-usd 3] [--detach]
//!   ti train modal status                                                              our apps + runs on the Volume
//!   ti train modal fetch  --tag T [--out dir]                                          pull /runs/T
//!   ti train modal stop   --id ap-...                                                  stop an app
use std::process::{Command, ExitCode, Stdio};

const VOLUME: &str = "ti-data";
const SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/modal/ti_modal.py");
/// Worst-case $/h of one container (A10G GPU ~$1.10 + 8 vCPU + 16 GiB). The 2026-10-31 bench ran far below this.
const RATE_USD_H: f64 = 1.6;

fn modal(args: &[&str]) -> Command {
    let mut c = Command::new("python");
    c.arg("-m").arg("modal").args(args).env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1");
    c
}

fn out(c: &mut Command) -> String {
    match c.stderr(Stdio::inherit()).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).into_owned(),
        Err(e) => format!("(failed to run: {e})"),
    }
}

fn hub_sha() -> String {
    let o = Command::new("git").args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "--short=8", "HEAD"]).output();
    o.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

pub fn run_modal(args: &[String]) -> ExitCode {
    let f = |n: &str| crate::flag_str(args, n);
    match args.first().map(String::as_str) {
        Some("put") => {
            let Some(files) = f("--files") else {
                eprintln!("usage: ti train modal put --files a,b,c [--dest data] [--force]");
                return ExitCode::FAILURE;
            };
            let dest = f("--dest").unwrap_or_else(|| "data".into());
            let listing = out(&mut modal(&["volume", "ls", VOLUME, &format!("/{dest}")]));
            for p in files.split(',').filter(|s| !s.is_empty()) {
                let name = std::path::Path::new(p).file_name().unwrap().to_string_lossy().into_owned();
                if listing.contains(&name) && !crate::has_flag(args, "--force") {
                    println!("[skip] {name} already on the volume (use --force to overwrite)");
                    continue;
                }
                println!("[put] {p} -> /{dest}/{name}");
                let st = modal(&["volume", "put", "-f", VOLUME, p, &format!("/{dest}/{name}")]).status();
                if !st.map(|s| s.success()).unwrap_or(false) {
                    eprintln!("put failed for {p}");
                    return ExitCode::FAILURE;
                }
            }
            ExitCode::SUCCESS
        }
        Some("launch") => {
            let (Some(tag), Some(run_args)) = (f("--tag"), f("--args")) else {
                eprintln!("usage: ti train modal launch --tag T --args \"train run --backend cuda --data /vol/data/train_v9 ...\" [--timeout-min 30] [--max-usd 3] [--detach]");
                return ExitCode::FAILURE;
            };
            let tmin = crate::flag_u64(args, "--timeout-min").unwrap_or(30);
            let max_usd = crate::flag_f64(args, "--max-usd").unwrap_or(3.0);
            let worst = tmin as f64 / 60.0 * RATE_USD_H;
            if worst > max_usd {
                eprintln!("refused: worst case ${worst:.2} ({tmin} min x ${RATE_USD_H}/h) > --max-usd {max_usd}");
                return ExitCode::FAILURE;
            }
            if !tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                eprintln!("refused: --tag must be [A-Za-z0-9_-]");
                return ExitCode::FAILURE;
            }
            println!("[launch] tag {tag}  hub {}  timeout {tmin} min  worst case ${worst:.2}", hub_sha());
            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let row = format!("{{\"ts\":{ts},\"tag\":{},\"hub\":\"{}\",\"timeout_min\":{tmin},\"worst_usd\":{worst:.2},\"args\":{}}}\n", crate::json_str(&tag), hub_sha(), crate::json_str(&run_args));
            let log = concat!(env!("CARGO_MANIFEST_DIR"), "/docs/MODAL_RUNS.jsonl");
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(log).and_then(|mut fh| std::io::Write::write_all(&mut fh, row.as_bytes()));
            let tmin_s = tmin.to_string();
            let mut base = vec!["run"];
            if crate::has_flag(args, "--detach") {
                base.push("--detach");
            }
            // the launcher has two local entrypoints (main = training, gen_main = generation): name the one we mean
            let entry = format!("{SCRIPT}::main");
            base.extend([entry.as_str(), "--tag", &tag, "--args", &run_args, "--timeout-min", &tmin_s]);
            match modal(&base).status() {
                Ok(s) if s.success() => ExitCode::SUCCESS,
                _ => ExitCode::FAILURE,
            }
        }
        Some("status") => {
            let apps = out(&mut modal(&["app", "list"]));
            for l in apps.lines().filter(|l| l.contains("ti-trai") || l.contains("App ID")) {
                println!("{l}");
            }
            println!("--- /runs on the volume");
            print!("{}", out(&mut modal(&["volume", "ls", VOLUME, "/runs"])));
            ExitCode::SUCCESS
        }
        Some("fetch") => {
            let Some(tag) = f("--tag") else {
                eprintln!("usage: ti train modal fetch --tag T [--out dir] [--all]");
                return ExitCode::FAILURE;
            };
            let dir = f("--out").unwrap_or_else(|| format!("runs/{tag}"));
            // `modal volume get --force` creates a FILE when the local dir is missing (AGENTS trap): make it first
            let _ = std::fs::create_dir_all(&dir);
            if crate::has_flag(args, "--all") {
                return match modal(&["volume", "get", "--force", VOLUME, &format!("/runs/{tag}"), &dir]).status() {
                    Ok(s) if s.success() => ExitCode::SUCCESS,
                    _ => ExitCode::FAILURE,
                };
            }
            // D819: a --keep-all run holds hundreds of 29 MB checkpoints (~10 GB); by default fetch only what export and
            // gating need (the checkpoints stay on the volume; --all downloads everything)
            let sub = format!("{dir}/{tag}");
            let _ = std::fs::create_dir_all(&sub);
            let mut got = 0;
            for name in ["best.tnn", "RECORD.txt", "EARLY_STOP.txt", "GUARD_TRIPPED.txt", "gpu_state.txt"] {
                let ok = modal(&["volume", "get", "--force", VOLUME, &format!("/runs/{tag}/{name}"), &format!("{sub}/{name}")])
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                if ok {
                    got += 1;
                } else if name == "best.tnn" || name == "RECORD.txt" {
                    eprintln!("fetch: {name} missing for run {tag}");
                    return ExitCode::FAILURE;
                }
            }
            println!("fetch: {got} files of run {tag} -> {sub} (checkpoints left on the volume; --all fetches them)");
            ExitCode::SUCCESS
        }
        Some("stop") => {
            let Some(id) = f("--id") else {
                eprintln!("usage: ti train modal stop --id ap-...");
                return ExitCode::FAILURE;
            };
            match modal(&["app", "stop", "-y", &id]).status() {
                Ok(s) if s.success() => ExitCode::SUCCESS,
                _ => ExitCode::FAILURE,
            }
        }
        _ => {
            eprintln!("usage: ti train modal put|launch|status|fetch|stop  (see src/modal.rs)");
            ExitCode::FAILURE
        }
    }
}
