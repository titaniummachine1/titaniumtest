//! `ti data pack`: v9 npz -> fixed-record `.tdat` (+ `.kata`, `.aux` so nothing is lost) + manifest.
//! Streams every column through its own deflate reader; memory stays at a few chunks.
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::process::ExitCode;

use flate2::read::DeflateDecoder;

pub const REC: usize = 48;
pub const KATA_REC: usize = 16;
pub const AUX_REC: usize = 12;

/// One `.tdat` record, decoded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rec {
    pub occ0: u64,
    pub occ1: u64,
    pub outcome: f32,
    pub cp: f32,
    pub wsum: f32,
    pub p3: f32,
    pub cp5k: f32,
    pub zx: i32,
    pub stm: u8,
    pub teacher_kind: u8,
    pub proof_kind: u8,
    pub relabel_src: u8,
    pub ti_clash: u8,
}

impl Rec {
    pub fn to_bytes(&self) -> [u8; REC] {
        let mut b = [0u8; REC];
        b[0..8].copy_from_slice(&self.occ0.to_le_bytes());
        b[8..16].copy_from_slice(&self.occ1.to_le_bytes());
        b[16..20].copy_from_slice(&self.outcome.to_le_bytes());
        b[20..24].copy_from_slice(&self.cp.to_le_bytes());
        b[24..28].copy_from_slice(&self.wsum.to_le_bytes());
        b[28..32].copy_from_slice(&self.p3.to_le_bytes());
        b[32..36].copy_from_slice(&self.cp5k.to_le_bytes());
        b[36..40].copy_from_slice(&self.zx.to_le_bytes());
        b[40] = self.stm;
        b[41] = self.teacher_kind;
        b[42] = self.proof_kind;
        b[43] = self.relabel_src;
        b[44] = self.ti_clash;
        b
    }
    pub fn from_bytes(b: &[u8]) -> Rec {
        let u = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let f = |i: usize| f32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        Rec {
            occ0: u(0),
            occ1: u(8),
            outcome: f(16),
            cp: f(20),
            wsum: f(24),
            p3: f(28),
            cp5k: f(32),
            zx: i32::from_le_bytes(b[36..40].try_into().unwrap()),
            stm: b[40],
            teacher_kind: b[41],
            proof_kind: b[42],
            relabel_src: b[43],
            ti_clash: b[44],
        }
    }
}

struct ZipEntry {
    name: String,
    method: u16,
    csize: u64,
    lho: u64,
}

fn rd_u16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes(b[i..i + 2].try_into().unwrap())
}
fn rd_u32(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}
fn rd_u64(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

fn zip_entries(path: &str) -> Result<Vec<ZipEntry>, String> {
    let mut f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let len = f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    let tail_n = len.min(1 << 16) as usize;
    f.seek(SeekFrom::Start(len - tail_n as u64)).map_err(|e| e.to_string())?;
    let mut tail = vec![0u8; tail_n];
    f.read_exact(&mut tail).map_err(|e| e.to_string())?;
    let eocd = (0..=tail_n.saturating_sub(22))
        .rev()
        .find(|&i| rd_u32(&tail, i) == 0x0605_4b50)
        .ok_or("zip: no end-of-central-directory")?;
    let mut count = rd_u16(&tail, eocd + 10) as u64;
    let mut cd_off = rd_u32(&tail, eocd + 16) as u64;
    if cd_off == 0xFFFF_FFFF || count == 0xFFFF {
        let loc = (0..eocd).rev().find(|&i| rd_u32(&tail, i) == 0x0706_4b50).ok_or("zip64 locator missing")?;
        let z64 = rd_u64(&tail, loc + 8);
        f.seek(SeekFrom::Start(z64)).map_err(|e| e.to_string())?;
        let mut h = [0u8; 56];
        f.read_exact(&mut h).map_err(|e| e.to_string())?;
        if rd_u32(&h, 0) != 0x0606_4b50 {
            return Err("zip64 eocd bad".into());
        }
        count = rd_u64(&h, 32);
        cd_off = rd_u64(&h, 48);
    }
    f.seek(SeekFrom::Start(cd_off)).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for _ in 0..count {
        let mut h = [0u8; 46];
        f.read_exact(&mut h).map_err(|e| e.to_string())?;
        if rd_u32(&h, 0) != 0x0201_4b50 {
            return Err("zip: bad central header".into());
        }
        let method = rd_u16(&h, 10);
        let mut csize = rd_u32(&h, 20) as u64;
        let mut usize_ = rd_u32(&h, 24) as u64;
        let (nl, el, cl) = (rd_u16(&h, 28) as usize, rd_u16(&h, 30) as usize, rd_u16(&h, 32) as usize);
        let mut lho = rd_u32(&h, 42) as u64;
        let mut name = vec![0u8; nl];
        f.read_exact(&mut name).map_err(|e| e.to_string())?;
        let mut extra = vec![0u8; el];
        f.read_exact(&mut extra).map_err(|e| e.to_string())?;
        let mut skip = vec![0u8; cl];
        f.read_exact(&mut skip).map_err(|e| e.to_string())?;
        let mut p = 0;
        while p + 4 <= extra.len() {
            let (id, sz) = (rd_u16(&extra, p), rd_u16(&extra, p + 2) as usize);
            if id == 1 {
                let mut q = p + 4;
                if usize_ == 0xFFFF_FFFF {
                    usize_ = rd_u64(&extra, q);
                    q += 8;
                }
                if csize == 0xFFFF_FFFF {
                    csize = rd_u64(&extra, q);
                    q += 8;
                }
                if lho == 0xFFFF_FFFF {
                    lho = rd_u64(&extra, q);
                }
            }
            p += 4 + sz;
        }
        let _ = usize_;
        out.push(ZipEntry { name: String::from_utf8_lossy(&name).into_owned(), method, csize, lho });
    }
    Ok(out)
}

/// Streaming reader of one `.npy` column inside the npz.
pub struct Col {
    rd: Box<dyn Read>,
    pub rows: usize,
    pub width: usize,
    pub elem: usize,
}

fn open_col(path: &str, e: &ZipEntry) -> Result<Col, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(e.lho)).map_err(|e| e.to_string())?;
    let mut lh = [0u8; 30];
    f.read_exact(&mut lh).map_err(|e| e.to_string())?;
    if rd_u32(&lh, 0) != 0x0403_4b50 {
        return Err("zip: bad local header".into());
    }
    let data = e.lho + 30 + rd_u16(&lh, 26) as u64 + rd_u16(&lh, 28) as u64;
    f.seek(SeekFrom::Start(data)).map_err(|e| e.to_string())?;
    let raw = BufReader::with_capacity(1 << 20, f).take(e.csize);
    let mut rd: Box<dyn Read> = match e.method {
        0 => Box::new(raw),
        8 => Box::new(DeflateDecoder::new(raw)),
        m => return Err(format!("zip method {m}")),
    };
    let mut m = [0u8; 10];
    rd.read_exact(&mut m).map_err(|e| e.to_string())?;
    if &m[0..6] != b"\x93NUMPY" {
        return Err(format!("{}: not npy", e.name));
    }
    if m[6] != 1 {
        return Err("npy v2+ unsupported".into());
    }
    let hl = rd_u16(&m, 8) as usize;
    let mut hdr = vec![0u8; hl];
    rd.read_exact(&mut hdr).map_err(|e| e.to_string())?;
    let h = String::from_utf8_lossy(&hdr).into_owned();
    if h.contains("'fortran_order': True") {
        return Err("fortran order unsupported".into());
    }
    let descr = h.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).ok_or("no descr")?;
    let elem = match descr {
        "<u8" | "<i8" => 8,
        "<f4" | "<i4" | "<u4" => 4,
        "<f2" | "<i2" | "<u2" => 2,
        "|u1" | "|i1" => 1,
        d => return Err(format!("dtype {d}")),
    };
    let shape = h
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or("no shape")?;
    let dims: Vec<usize> = shape.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    let rows = *dims.first().ok_or("empty shape")?;
    // D517: N-d arrays (Kata tdata: binaryInputNCHWPacked N x 22 x 7, policyTargetsNCMove N x 2 x 50): width = product of the rest
    let width = dims[1..].iter().product::<usize>().max(1);
    Ok(Col { rd, rows, width, elem })
}

impl Col {
    fn row_bytes(&self) -> usize {
        self.width * self.elem
    }
    pub fn read(&mut self, n: usize, buf: &mut Vec<u8>) -> Result<(), String> {
        buf.resize(n * self.row_bytes(), 0);
        self.rd.read_exact(buf).map_err(|e| e.to_string())
    }
}

pub struct Npz {
    path: String,
    entries: Vec<ZipEntry>,
}

impl Npz {
    pub fn open(path: &str) -> Result<Npz, String> {
        Ok(Npz { path: path.to_string(), entries: zip_entries(path)? })
    }
    pub fn col(&self, key: &str) -> Result<Col, String> {
        let want = format!("{key}.npy");
        // D517: KataGo tdata names its members without the .npy suffix
        let e = self.entries.iter().find(|e| e.name == want || e.name == key).ok_or(format!("column {key} missing"))?;
        open_col(&self.path, e)
    }
}

const CHUNK: usize = 1 << 18;

fn wsum_u8(b: &[u8]) -> u64 {
    b.iter().fold(0u64, |a, &x| a.wrapping_add(x as u64))
}
fn wsum_u64(b: &[u8]) -> u64 {
    b.chunks_exact(8).fold(0u64, |a, c| a.wrapping_add(u64::from_le_bytes(c.try_into().unwrap())))
}
fn wsum_u32(b: &[u8]) -> u64 {
    b.chunks_exact(4).fold(0u64, |a, c| a.wrapping_add(u32::from_le_bytes(c.try_into().unwrap()) as u64))
}

/// Column order after occ0/occ1; indexes into `bufs`.
const NAMES: [&str; 16] = [
    "outcome_p", "titanium_cp", "wsum", "teacher_p3", "titanium_cp_5k", "zx_score", // 0..6 (4 B each)
    "stm", "teacher_kind", "proof_kind", "relabel_src", "ti_clash", // 6..11 (1 B each)
    "kata_td_win", "kata_td_draw", // 11,12 (8 B per row)
    "titanium_cp_old", "titanium_cp_v8", "zx_nodes", // 13..16 (4 B each)
];

/// Pack one npz. Returns (rows, manifest json). `wrapping_sums` are per-column sums mod 2^64 so a
/// reader can prove the pack equals the source (python: `int(col.astype(np.uint64).sum())` per dtype view).
pub fn pack(npz_path: &str, out_stem: &str) -> Result<(usize, String), String> {
    let z = Npz::open(npz_path)?;
    let mut c_occ0 = z.col("occ0")?;
    let mut c_occ1 = z.col("occ1")?;
    let rows = c_occ0.rows;
    let mut cols: Vec<Col> = Vec::new();
    for n in NAMES {
        let c = z.col(n)?;
        if c.rows != rows {
            return Err(format!("{n}: {} rows != {rows}", c.rows));
        }
        cols.push(c);
    }
    let mk = |ext: &str| -> Result<BufWriter<File>, String> {
        Ok(BufWriter::new(File::create(format!("{out_stem}.{ext}")).map_err(|e| e.to_string())?))
    };
    let (mut w_t, mut w_k, mut w_a) = (mk("tdat")?, mk("kata")?, mk("aux")?);
    let mut sums = [0u64; 18];
    let (mut b0, mut b1) = (Vec::new(), Vec::new());
    let mut bufs: Vec<Vec<u8>> = vec![Vec::new(); cols.len()];
    let mut done = 0;
    while done < rows {
        let n = CHUNK.min(rows - done);
        c_occ0.read(n, &mut b0)?;
        c_occ1.read(n, &mut b1)?;
        for (c, b) in cols.iter_mut().zip(bufs.iter_mut()) {
            c.read(n, b)?;
        }
        sums[0] = sums[0].wrapping_add(wsum_u64(&b0));
        sums[1] = sums[1].wrapping_add(wsum_u64(&b1));
        for k in (0..6).chain(13..16) {
            sums[2 + k] = sums[2 + k].wrapping_add(wsum_u32(&bufs[k]));
        }
        for k in 6..11 {
            sums[2 + k] = sums[2 + k].wrapping_add(wsum_u8(&bufs[k]));
        }
        for i in 0..n {
            let f4 = |k: usize| f32::from_le_bytes(bufs[k][i * 4..i * 4 + 4].try_into().unwrap());
            let r = Rec {
                occ0: u64::from_le_bytes(b0[i * 8..i * 8 + 8].try_into().unwrap()),
                occ1: u64::from_le_bytes(b1[i * 8..i * 8 + 8].try_into().unwrap()),
                outcome: f4(0),
                cp: f4(1),
                wsum: f4(2),
                p3: f4(3),
                cp5k: f4(4),
                zx: i32::from_le_bytes(bufs[5][i * 4..i * 4 + 4].try_into().unwrap()),
                stm: bufs[6][i],
                teacher_kind: bufs[7][i],
                proof_kind: bufs[8][i],
                relabel_src: bufs[9][i],
                ti_clash: bufs[10][i],
            };
            w_t.write_all(&r.to_bytes()).map_err(|e| e.to_string())?;
            w_k.write_all(&bufs[11][i * 8..i * 8 + 8]).map_err(|e| e.to_string())?;
            w_k.write_all(&bufs[12][i * 8..i * 8 + 8]).map_err(|e| e.to_string())?;
            for k in 13..16 {
                w_a.write_all(&bufs[k][i * 4..i * 4 + 4]).map_err(|e| e.to_string())?;
            }
        }
        done += n;
    }
    for w in [&mut w_t, &mut w_k, &mut w_a] {
        w.flush().map_err(|e| e.to_string())?;
    }
    let mut keys: Vec<&str> = vec!["occ0", "occ1"];
    keys.extend(NAMES.iter());
    let body: Vec<String> = keys
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 13 && *i != 14)
        .map(|(i, k)| format!("\"{k}\":{}", sums[i]))
        .collect();
    let json = format!(
        "{{\"src\":\"{}\",\"rows\":{rows},\"rec\":{REC},\"kata_rec\":{KATA_REC},\"aux_rec\":{AUX_REC},\"wrapping_sums\":{{{}}}}}",
        npz_path.replace('\\', "/"),
        body.join(",")
    );
    std::fs::write(format!("{out_stem}.manifest.json"), &json).map_err(|e| e.to_string())?;
    Ok((rows, json))
}

pub fn run_data(args: &[String]) -> ExitCode {
    match args.first().map(String::as_str) {
        Some("pack") => {
            let (Some(src), Some(out)) = (crate::flag_str(args, "--npz"), crate::flag_str(args, "--out")) else {
                eprintln!("usage: ti data pack --npz <v9.npz> --out <stem>   (writes stem.tdat/.kata/.aux/.manifest.json)");
                return ExitCode::FAILURE;
            };
            let t = std::time::Instant::now();
            match pack(&src, &out) {
                Ok((rows, json)) => {
                    println!("packed {rows} rows in {:.1}s -> {out}.tdat ({} B)", t.elapsed().as_secs_f64(), rows * REC);
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("data pack: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("check-targets") => crate::targets::run_check(args),
        Some("check-leak") => crate::leak::run_check(args),
        Some("stats") => run_stats(args),
        _ => {
            eprintln!("usage: ti data pack --npz <v9.npz> --out <stem>");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rec_roundtrip() {
        let r = Rec { occ0: 0x1234_5678_9abc_def0, occ1: 7, outcome: 0.5, cp: -123.5, wsum: 2.0, p3: 0.25, cp5k: 9.0, zx: -31990, stm: 1, teacher_kind: 2, proof_kind: 3, relabel_src: 2, ti_clash: 1 };
        assert_eq!(Rec::from_bytes(&r.to_bytes()), r);
    }
    #[test]
    fn val_v9_packs_when_present() {
        let p = "C:/gitProjects/titanium-ataxx/data/nnue/modal_up/final_val_v9.npz";
        if !std::path::Path::new(p).exists() {
            return;
        }
        let out = std::env::temp_dir().join("ti_pack_test");
        let (rows, _) = pack(p, out.to_str().unwrap()).unwrap();
        assert_eq!(rows, 601_957);
        assert_eq!(std::fs::metadata(format!("{}.tdat", out.display())).unwrap().len(), (rows * REC) as u64);
    }
}

/// `ti data stats --data <stem>`: what is in a packed dataset (source of the labels, how many carry a 5k-search eval, proofs).
pub fn run_stats(args: &[String]) -> ExitCode {
    use std::collections::BTreeMap;
    let Some(stem) = crate::flag_str(args, "--data") else {
        eprintln!("usage: ti data stats --data <stem>");
        return ExitCode::FAILURE;
    };
    let mut t = match crate::trainer::Tdat::open(&stem) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("data stats: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut by: BTreeMap<(u8, u8, u8), [u64; 5]> = BTreeMap::new(); // rows, proven, has_cp5k, zx!=0, ti_clash
    let (mut buf, mut s) = (Vec::new(), 0usize);
    let (mut sum_gap, mut n_gap) = (0f64, 0u64);
    let mut emp = [[0u64; 3]; 50]; // per empties: rows, proof-labelled, 5k search found a forced result
    while s < t.rows {
        buf.clear();
        if t.read_block(s, 1 << 16, &mut buf).is_err() {
            break;
        }
        s += buf.len();
        for r in &buf {
            let ne = (crate::mapper::board_of(r, 0).empty().count_ones() as usize).min(49);
            emp[ne][0] += 1;
            if r.cp.abs() >= 98_000.0 {
                emp[ne][1] += 1;
            }
            if r.cp5k.abs() >= 98_000.0 {
                emp[ne][2] += 1;
            }
            let e = by.entry((r.teacher_kind, r.proof_kind, r.relabel_src)).or_default();
            e[0] += 1;
            if r.cp.abs() >= 98_000.0 {
                e[1] += 1;
            }
            if r.cp5k != 0.0 {
                e[2] += 1;
                if r.cp.abs() < 98_000.0 {
                    sum_gap += (r.cp - r.cp5k).abs().min(1500.0) as f64;
                    n_gap += 1;
                }
            }
            if r.zx != 0 {
                e[3] += 1;
            }
            if r.ti_clash != 0 {
                e[4] += 1;
            }
        }
    }
    println!("{stem}: {} rows", t.rows);
    println!("teacher_kind proof_kind relabel_src |       rows   proven(|cp|>=98000)   cp5k!=0     zx!=0  ti_clash");
    let mut tot = [0u64; 5];
    for ((a, b, c), e) in &by {
        println!("{a:>12} {b:>10} {c:>11} | {:>10} {:>10} {:>14} {:>9} {:>9}", e[0], e[1], e[2], e[3], e[4]);
        for i in 0..5 {
            tot[i] += e[i];
        }
    }
    println!("total {} rows: proven {} ({:.1}%), cp5k present {} ({:.1}%), zx {}, clash {}; mean |cp - cp5k| (capped 1500) on non-proven rows with both: {:.1} cp", tot[0], tot[1], 100.0 * tot[1] as f64 / tot[0] as f64, tot[2], 100.0 * tot[2] as f64 / tot[0] as f64, tot[3], tot[4], sum_gap / n_gap.max(1) as f64);
    println!("empties |       rows  proof-labelled (%)   5k-search-found-forced (%)");
    for (ne, e) in emp.iter().enumerate().filter(|(_, e)| e[0] > 0) {
        println!("{ne:>7} | {:>10} {:>10} ({:>5.1})  {:>10} ({:>5.1})", e[0], e[1], 100.0 * e[1] as f64 / e[0] as f64, e[2], 100.0 * e[2] as f64 / e[0] as f64);
    }
    ExitCode::SUCCESS
}
