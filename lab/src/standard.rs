//! Lab training standard (user order D1130: "remove any way to disobey me accidentally").
//!
//! The standard is DATA (`lab/STANDARD_TRAIN.txt`): the whole dataset, its exact row count, the held-out tail and the
//! champion target. Every lab trainer calls [`train_split`]: without `--experiment NAME` it refuses every flag that
//! could train on less data or another target (`--data` other than the standard file, `--max-rows`, `--holdout`,
//! `--val-split`, `--val-target`, `--val`), loads ALL rows and holds out exactly the standard tail. With
//! `--experiment NAME` the old flags work, but the output must sit under a folder named `experiments` and every log
//! and model says EXPERIMENT, so an experiment can never be mistaken for a standard result.
use crate::hcetrain::read_rows;
use crate::util::{arg, Rec, REC};

pub const FILE: &str = "C:/gitProjects/titaniumtest/lab/STANDARD_TRAIN.txt";

pub struct Std {
    pub data: String,
    pub rows: usize,
    pub holdout: usize,
}

pub fn load() -> Result<Std, String> {
    let s = std::fs::read_to_string(FILE).map_err(|e| format!("{FILE}: {e} (the lab training standard is missing)"))?;
    let get = |k: &str| -> Result<String, String> {
        s.lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .find_map(|l| l.split_once('=').filter(|(a, _)| a.trim() == k).map(|(_, b)| b.trim().to_string()))
            .ok_or(format!("{FILE}: no '{k} =' line"))
    };
    let num = |k: &str| -> Result<usize, String> { get(k)?.parse::<usize>().map_err(|e| format!("{FILE}: {k}: {e}")) };
    if get("target")? != "champion" {
        return Err(format!("{FILE}: target must be 'champion'"));
    }
    Ok(Std { data: get("data")?, rows: num("rows")?, holdout: num("holdout_rows")? })
}

/// `Some(name)` when the caller explicitly asked for an experiment.
pub fn experiment(args: &[String]) -> Option<String> {
    arg(args, "--experiment").filter(|s| !s.trim().is_empty())
}

fn same_path(a: &str, b: &str) -> bool {
    let n = |p: &str| {
        let p = std::fs::canonicalize(p).map(|x| x.to_string_lossy().to_string()).unwrap_or_else(|_| p.to_string());
        p.replace('\\', "/").trim_start_matches("//?/").to_lowercase()
    };
    n(a) == n(b)
}

/// Refuses an experiment whose output is not under an `experiments` folder and a standard run whose output is.
pub fn check_out(out: &str, exp: &Option<String>) -> Result<(), String> {
    let under = out.replace('\\', "/").split('/').any(|c| c.eq_ignore_ascii_case("experiments"));
    match (exp, under) {
        (Some(n), false) => Err(format!("--experiment {n}: --out must be under a folder named 'experiments' (got {out})")),
        (None, true) => Err(format!("--out {out} is under 'experiments' but this is a standard run; pass --experiment NAME or move it")),
        _ => Ok(()),
    }
}

/// The one line every log starts with.
pub fn stamp(exp: &Option<String>, st: &Std, train: usize, val: usize) -> String {
    match exp {
        None => format!(
            "STANDARD (lab/STANDARD_TRAIN.txt, D1130): WHOLE dataset {} = {} rows: {train} train rows + {val} held-out rows, champion target",
            st.data, st.rows
        ),
        Some(n) => format!("EXPERIMENT {n}: NOT the standard (whole dataset / champion target not guaranteed): {train} train rows, {val} val rows"),
    }
}

/// The standard split, or refuses. `forbidden` = flags that bend the data or the target in this tool.
pub fn standard_split(args: &[String], forbidden: &[&str]) -> Result<(Std, Vec<Rec>, Vec<Rec>), String> {
    let st = load()?;
    for f in forbidden {
        if args.iter().any(|a| a == f) {
            return Err(format!(
                "{f} is refused: the standard trains on the WHOLE dataset with the champion target and holds out the last {} rows (lab/STANDARD_TRAIN.txt, user order D1130). Use --experiment NAME (output under experiments/) only for a deliberate non-standard run.",
                st.holdout
            ));
        }
    }
    if let Some(d) = arg(args, "--data") {
        if !same_path(&d, &st.data) {
            return Err(format!("--data {d} is not the standard dataset {} (D1130); a subset needs --experiment NAME", st.data));
        }
    }
    let len = std::fs::metadata(&st.data).map_err(|e| format!("{}: {e}", st.data))?.len() as usize;
    if len % REC != 0 || len / REC != st.rows {
        return Err(format!(
            "{} has {} rows, the standard says {} (file changed): update lab/STANDARD_TRAIN.txt only by the user's order",
            st.data,
            len / REC,
            st.rows
        ));
    }
    let mut all = read_rows(&st.data, usize::MAX);
    if all.len() != st.rows {
        return Err(format!("read {} rows of {}, expected {}", all.len(), st.data, st.rows));
    }
    let val = all.split_off(st.rows - st.holdout);
    Ok((st, all, val))
}
