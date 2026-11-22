//! LAWS A14 (user D1305-D1307): every job stops SAFELY when asked and keeps its progress.
//!
//! Stop files (either one stops the job):
//!   * `<run dir>/STOP`            -- stop this one run (the run's output / save folder)
//!   * `C:/gitProjects/STOP_ALL`   -- stop everything (`docs/analysis/stop_all.ps1` creates it, waits, then removes it);
//!     override the path with the environment variable `TI_STOP_ALL` (tests).
//! Trainers save their best weights + a full checkpoint and exit; gates start no new pair, keep every finished game and
//! still write their summary; generators start no new game. Checkpoints are written atomically (tmp + rename), so a power
//! cut during a write never leaves a broken file.
use std::path::{Path, PathBuf};

pub const STOP_ALL_DEFAULT: &str = "C:/gitProjects/STOP_ALL";

pub fn stop_all_path() -> PathBuf {
    std::env::var("TI_STOP_ALL").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(STOP_ALL_DEFAULT))
}

/// Some(file that asked for the stop) when the job must stop now.
pub fn requested(run_dir: Option<&Path>) -> Option<String> {
    let all = stop_all_path();
    if all.exists() {
        return Some(all.display().to_string());
    }
    if let Some(d) = run_dir {
        let p = d.join("STOP");
        if p.exists() {
            return Some(p.display().to_string());
        }
    }
    // a parent job (e.g. `ti spsa`) passes its own run folder to the child processes it starts
    if let Ok(d) = std::env::var("TI_STOP_DIR") {
        let p = Path::new(&d).join("STOP");
        if p.exists() {
            return Some(p.display().to_string());
        }
    }
    None
}

/// Write `bytes` to `path` atomically: a power cut leaves either the old file or the new one, never a torn one.
pub fn write_atomic(path: &str, bytes: &[u8]) -> Result<(), String> {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, bytes).map_err(|e| format!("{tmp}: {e}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{tmp} -> {path}: {e}"))
}

/// Little-endian byte writer / reader for checkpoints.
#[derive(Default)]
pub struct Wr(pub Vec<u8>);
impl Wr {
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn f32s(&mut self, v: &[f32]) -> &mut Self {
        self.u64(v.len() as u64);
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
        self
    }
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
        self
    }
}

pub struct Rd<'a> {
    pub d: &'a [u8],
    pub p: usize,
}
impl<'a> Rd<'a> {
    pub fn new(d: &'a [u8]) -> Rd<'a> {
        Rd { d, p: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.d.len() {
            return Err("checkpoint truncated".into());
        }
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f32s(&mut self) -> Result<Vec<f32>, String> {
        let n = self.u64()? as usize;
        Ok(self.take(4 * n)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    pub fn str(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_files_and_atomic_roundtrip() {
        let dir = std::env::temp_dir().join(format!("stopctl_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("TI_STOP_ALL", dir.join("STOP_ALL_TEST"));
        assert!(requested(Some(&dir)).is_none());
        std::fs::write(dir.join("STOP"), b"").unwrap();
        assert!(requested(Some(&dir)).unwrap().ends_with("STOP"));
        std::fs::remove_file(dir.join("STOP")).unwrap();
        std::fs::write(dir.join("STOP_ALL_TEST"), b"").unwrap();
        assert!(requested(None).unwrap().ends_with("STOP_ALL_TEST"));
        std::fs::remove_file(dir.join("STOP_ALL_TEST")).unwrap();
        let f = dir.join("ck.bin");
        let fp = f.to_str().unwrap();
        let mut w = Wr::default();
        w.u64(7).f64(-1.5).f32s(&[1.0, 2.5]).str("abc");
        write_atomic(fp, &w.0).unwrap();
        let b = std::fs::read(fp).unwrap();
        let mut r = Rd::new(&b);
        assert_eq!((r.u64().unwrap(), r.f64().unwrap(), r.f32s().unwrap(), r.str().unwrap()), (7, -1.5, vec![1.0, 2.5], "abc".to_string()));
        assert!(!Path::new(&format!("{fp}.tmp")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
