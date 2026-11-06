//! Reproducibility guard for every gate (engine D416, user 2026-11-06: "idiot-proof it so you can always replicate what and
//! how was run; only gated commits, committed to GitHub on the experiment branch").
//!
//! What it enforces in `ti sprt` (and therefore `ti gate`, which calls it):
//! 1. A gate vs `main` may only play a candidate exe that `ti gate` built from a commit (`<engine>/../_gate/exes/titanium-<sha8>.exe`).
//!    A raw `target/release/...` exe is refused unless `--dev-raw-exe`, which stamps the log `DEV_RAW_EXE (NOT a result)`.
//! 2. Every file argument in either engine spec (models, nets, books) is hashed into the header: `file <name>=<SHA8>`, so an
//!    overwritten model can never hide behind its name again (the 2026-11-06 v24 case).
//! 3. `--tstg` without an explicit file is refused (the engine's built-in default path changed four times in two days).
//! 4. Every run is appended to `<engine>/docs/GATE_RUNS.tsv` (tracked) (START row with the full header, DONE row with the result).
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::hash;

/// Hash label of every token after the exe that names an existing file (as given, or relative to the engine root).
pub fn file_args_label(spec: &str, root: Option<&Path>) -> String {
    let mut out = String::new();
    for t in spec.split_whitespace().skip(1) {
        if t.starts_with('-') {
            continue;
        }
        let p = PathBuf::from(t);
        let found = if p.is_file() { Some(p) } else { root.map(|r| r.join(t)).filter(|q| q.is_file()) };
        if let Some(f) = found {
            let name = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            out.push_str(&format!(" file {name}={}", hash::file_label(&f.to_string_lossy())));
        }
    }
    out
}

/// `--tstg` (or `--tstg` followed by another flag) means "whatever default the binary has": refused.
pub fn check_explicit_models(spec: &str) -> Result<(), String> {
    let toks: Vec<&str> = spec.split_whitespace().collect();
    for (i, t) in toks.iter().enumerate() {
        if *t == "--tstg" && toks.get(i + 1).map_or(true, |n| n.starts_with("--")) {
            return Err(format!(
                "engine spec `{spec}` uses `--tstg` without a model file: name the .ttst explicitly (the built-in default moved 4 times, D414)"
            ));
        }
    }
    Ok(())
}

/// True when the exe was built by `ti gate` from a commit (`_gate/exes/titanium-<sha8>.exe`).
pub fn is_gate_built(exe: &str) -> bool {
    let e = exe.replace('\\', "/").to_ascii_lowercase();
    e.contains("/_gate/exes/titanium-")
}

pub fn first_token(spec: &str) -> &str {
    spec.split_whitespace().next().unwrap_or("")
}

/// Append one row to `<engine>/docs/GATE_RUNS.tsv` (tracked) (never fails the gate; prints a warning instead).
pub fn log_run(root: Option<&Path>, status: &str, out: &str, text: &str) {
    let Some(r) = root else { return };
    let path = r.join("docs").join("GATE_RUNS.tsv");
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    let new = !path.exists();
    match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if new {
                let _ = writeln!(f, "utc\tstatus\tlog\tdetail");
            }
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let _ = writeln!(f, "{}\t{status}\t{out}\t{}", crate::gate::iso_utc(now), text.replace(['\t', '\n'], " "));
        }
        Err(e) => eprintln!("sprt: WARNING could not append {}: {e}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tstg_without_file_is_refused() {
        assert!(check_explicit_models("x.exe serve --tstg").is_err());
        assert!(check_explicit_models("x.exe serve --tstg --no-fut-sn").is_err());
        assert!(check_explicit_models("x.exe serve --tstg data/a.ttst").is_ok());
        assert!(check_explicit_models("x.exe serve").is_ok());
    }

    #[test]
    fn gate_built_detection() {
        assert!(is_gate_built("C:/gitProjects/_gate/exes/titanium-2c40ae5e.exe"));
        assert!(is_gate_built("C:\\gitProjects\\_gate\\exes\\titanium-2c40ae5e.exe"));
        assert!(!is_gate_built("C:/gitProjects/titanium-ataxx/target/release/titanium-cli.exe"));
    }
}
