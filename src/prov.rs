//! Net provenance (D242: a mislabelled local file, val 0.0206 instead of the run's 0.0178, produced a false -102 verdict
//! that steered a day of decisions). Three machine checks:
//! 1. the trainer appends every checkpoint it writes to `RECORD.txt` in the run dir (file, hash, val, val slice);
//! 2. `ti train export` re-measures the source net's val on that same slice and refuses on a mismatch; it writes
//!    `<out>.prov` (source hash, record line, val of source and output);
//! 3. `ti sprt` refuses a candidate net without a `.prov` whose hash matches the file (unless --allow-unproven-net).
use crate::hash;

pub const RECORD: &str = "RECORD.txt";

/// Basename of a tdat stem without the extension (the val file is found by name on another machine).
pub fn stem_name(stem: &str) -> String {
    let base = std::path::Path::new(stem).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    base.trim_end_matches(".tdat").to_string()
}

/// Append `<file> <sha8> val <mse> rows <n> on <val-stem> sb <superbatch>` to the run's RECORD.txt.
pub fn record(dir: &str, file: &str, bytes: &[u8], mse: f64, val_stem: &str, rows: usize, sb: usize) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/{RECORD}")) {
        let _ = writeln!(f, "{file} {} val {mse:.8} rows {rows} on {} sb {sb}", hash::short_hex(bytes), stem_name(val_stem));
    }
}

pub struct Entry {
    pub line: String,
    pub val: f64,
    pub rows: usize,
    pub on: String,
}

/// The RECORD.txt line (in the net's own dir) whose hash equals the net's hash, so renamed copies still match.
pub fn lookup(net: &str) -> Option<Entry> {
    let sha = hash::short_hex(&std::fs::read(net).ok()?);
    let dir = std::path::Path::new(net).parent()?;
    let text = std::fs::read_to_string(dir.join(RECORD)).ok()?;
    for l in text.lines().rev() {
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.len() >= 9 && t[1] == sha {
            return Some(Entry { line: l.to_string(), val: t[3].parse().ok()?, rows: t[5].parse().ok()?, on: t[7].to_string() });
        }
    }
    None
}

/// `.prov` of an exported net: `key value` lines, `out_sha` first.
pub fn write_prov(out: &str, lines: &[(&str, String)]) -> Result<(), String> {
    let body: String = lines.iter().map(|(k, v)| format!("{k} {v}\n")).collect();
    std::fs::write(format!("{out}.prov"), body).map_err(|e| format!("{out}.prov: {e}"))
}

/// Gate check for one net file: Ok(summary) if `<net>.prov` exists and its out_sha is the file's hash.
pub fn check_net(net: &str) -> Result<String, String> {
    let sha = hash::file_label(net);
    let text = std::fs::read_to_string(format!("{net}.prov")).map_err(|_| format!("{net} (sha {sha}) has no .prov: export it with `ti train export` (which verifies it against the run's RECORD.txt)"))?;
    let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(k).map(|v| v.trim().to_string()));
    let want = get("out_sha ").unwrap_or_default();
    if want != sha {
        return Err(format!("{net}: file sha {sha} != .prov out_sha {want} (file changed after export)"));
    }
    Ok(format!("{sha} from {} val {}", get("source_sha ").unwrap_or_default(), get("val_out ").unwrap_or_default()))
}

/// Every `.tnn` token of an engine spec (a bare net path or `exe ... --network x.tnn`).
pub fn nets_of(spec: &str) -> Vec<String> {
    let path = spec.strip_prefix("tnn:").unwrap_or(spec);
    if path.ends_with(".tnn") && !path.contains(char::is_whitespace) {
        return vec![path.to_string()];
    }
    spec.split_whitespace().filter(|t| t.ends_with(".tnn")).map(str::to_string).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_lookup_roundtrip_and_prov_check() {
        let dir = std::env::temp_dir().join(format!("ti_prov_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_string_lossy().to_string();
        let net = format!("{d}/best.tnn");
        std::fs::write(&net, b"net bytes").unwrap();
        record(&d, "best.tnn", b"net bytes", 0.0177970, "/vol/data/val_v9_clean", 131072, 13);
        let e = lookup(&net).expect("record found");
        assert!((e.val - 0.017797).abs() < 1e-9 && e.rows == 131072 && e.on == "val_v9_clean");
        // a renamed copy with other bytes does not match
        std::fs::write(format!("{d}/other.tnn"), b"other").unwrap();
        assert!(lookup(&format!("{d}/other.tnn")).is_none());
        assert!(check_net(&net).is_err());
        write_prov(&net, &[("out_sha", hash::file_label(&net)), ("source_sha", "X".into()), ("val_out", "0.0178".into())]).unwrap();
        assert!(check_net(&net).is_ok());
        std::fs::write(&net, b"changed").unwrap();
        assert!(check_net(&net).is_err());
        assert_eq!(nets_of("C:/a/ti.exe serve --network C:/n/a.tnn"), vec!["C:/n/a.tnn"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
