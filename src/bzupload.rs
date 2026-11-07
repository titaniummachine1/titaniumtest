//! `ti botzone-upload`: upload the Botzone package (botzone/upload/ from tools/build_botzone.py) from this PC.
//!
//! It asks for the Botzone login and password in the terminal (password hidden). Neither is stored, written to a file, or
//! placed on a command line: curl.exe reads them from its stdin (`-K -`), the cookie jar lives in the temp dir and is deleted
//! at the end (LAWS A6/17a). Steps: check the package against its MANIFEST.txt, log in, upload data/titanium_v2.so to
//! storage (curl works where `requests` times out, AGENTS sec. 7), download it back and compare sha256, switch the bot to
//! keep-running, create the new bot version from titanium_botzone.py.
//!
//!   ti botzone-upload [--dir DIR] [--desc TEXT (<=100 chars)] [--email ADDRESS] [--no-version] [--dry-run]
//!
//! --dry-run checks the package and prints what would happen, with no prompt and no network.

use crate::hash::sha256;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

const BASE: &str = "https://www.botzone.org.cn";
const BOT_ID: &str = "6ad7e0c3bdad901f323f6516"; // titanium_ataxx_PY6
const BOT_NAME: &str = "titanium_ataxx_PY6";
const GAME_ID: &str = "5809c7647f65182b044b5e3b"; // Ataxx

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn urlenc(s: &str) -> String {
    s.bytes()
        .map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") })
        .collect()
}

#[cfg(windows)]
fn read_hidden(prompt: &str) -> String {
    extern "system" {
        fn GetStdHandle(n: u32) -> isize;
        fn GetConsoleMode(h: isize, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: isize, mode: u32) -> i32;
    }
    eprint!("{prompt}");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    unsafe {
        let h = GetStdHandle(-10i32 as u32);
        let mut mode = 0u32;
        let hidden = GetConsoleMode(h, &mut mode) != 0 && SetConsoleMode(h, mode & !0x4) != 0;
        let _ = std::io::stdin().lock().read_line(&mut line);
        if hidden {
            SetConsoleMode(h, mode);
            eprintln!();
        } else {
            eprintln!("(input was visible: stdin is not a console)");
        }
    }
    line.trim_end_matches(['\r', '\n']).to_string()
}

#[cfg(not(windows))]
fn read_hidden(prompt: &str) -> String {
    eprint!("{prompt}(visible on this platform) ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    let _ = std::io::stdin().lock().read_line(&mut line);
    line.trim_end_matches(['\r', '\n']).to_string()
}

fn read_line(prompt: &str) -> String {
    eprint!("{prompt}");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    let _ = std::io::stdin().lock().read_line(&mut line);
    line.trim().to_string()
}

/// Removes the cookie jar when the command ends, however it ends.
struct Jar(PathBuf);
impl Drop for Jar {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// curl.exe with its config (URL, secrets) on stdin; returns stdout.
fn curl(config: &str, args: &[&str], show_progress: bool) -> Result<String, String> {
    let mut cmd = Command::new("curl.exe");
    cmd.args(["-sS", "-K", "-"]).args(args).stdin(Stdio::piped()).stdout(Stdio::piped());
    cmd.stderr(if show_progress { Stdio::inherit() } else { Stdio::piped() });
    let mut child = cmd.spawn().map_err(|e| format!("curl.exe not runnable: {e}"))?;
    child.stdin.take().unwrap().write_all(config.as_bytes()).map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("curl failed ({}): {}", out.status, String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Check every file listed in MANIFEST.txt ("  <sha256>  <size>  <path>") against the package folder.
fn check_manifest(dir: &Path) -> Result<String, String> {
    let m = std::fs::read_to_string(dir.join("MANIFEST.txt")).map_err(|e| format!("MANIFEST.txt: {e}"))?;
    let head = m.lines().next().unwrap_or("").to_string();
    let mut n = 0;
    for l in m.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.len() == 3 && f[0].len() == 64 && f[1].parse::<u64>().is_ok() {
            let d = std::fs::read(dir.join(f[2])).map_err(|e| format!("{}: {e}", f[2]))?;
            if hex(&sha256(&d)) != f[0] || d.len() as u64 != f[1].parse::<u64>().unwrap() {
                return Err(format!("{} does not match MANIFEST.txt (rebuild the package or fix the manifest)", f[2]));
            }
            n += 1;
        }
    }
    if n < 2 {
        return Err("MANIFEST.txt lists fewer than 2 files".into());
    }
    Ok(head)
}

pub fn run(args: &[String]) -> ExitCode {
    match go(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("botzone-upload: {e}");
            ExitCode::FAILURE
        }
    }
}

fn go(args: &[String]) -> Result<(), String> {
    let dir = PathBuf::from(flag(args, "--dir").unwrap_or_else(|| "C:/gitProjects/mainpromo-ataxx/botzone/upload".into()));
    let dry = args.iter().any(|a| a == "--dry-run");
    let no_version = args.iter().any(|a| a == "--no-version");
    let head = check_manifest(&dir)?;
    let so_path = dir.join("data").join("titanium_v2.so");
    let py_path = dir.join("titanium_botzone.py");
    let so = std::fs::read(&so_path).map_err(|e| format!("{}: {e}", so_path.display()))?;
    let py = std::fs::read(&py_path).map_err(|e| format!("{}: {e}", py_path.display()))?;
    if py.contains(&b'\r') {
        return Err("titanium_botzone.py has CRLF line endings: replace it with `git cat-file blob HEAD:botzone/titanium_botzone.py`".into());
    }
    let so_sha = hex(&sha256(&so));
    let commit = head.rsplit('@').next().unwrap_or("?").trim().to_string();
    // D531: no hard-coded version text: the default names the commit; the nets are the ones pinned in BUNDLE.txt there
    let desc = flag(args, "--desc").unwrap_or_else(|| format!("main {commit} (bundled net + policy + SmallNet, data/nnue/BUNDLE.txt)"));
    if desc.len() > 100 || desc.contains(['"', ';', '\\']) {
        return Err("--desc must be <= 100 chars and contain no quote, semicolon or backslash".into());
    }
    eprintln!("package   {}\n  {head}\n  titanium_v2.so {} B sha256 {}\n  driver {} B sha256 {}\nbot       {BOT_NAME} ({BOT_ID})\nversion   {}",
        dir.display(), so.len(), &so_sha[..16], py.len(), &hex(&sha256(&py))[..16], if no_version { "NOT created (--no-version)".to_string() } else { format!("\"{desc}\"") });
    if dry {
        eprintln!("dry run: package consistent, nothing sent");
        return Ok(());
    }
    let email = flag(args, "--email").unwrap_or_else(|| read_line("Botzone login (email): "));
    let password = read_hidden("Botzone password (hidden): ");
    if email.is_empty() || password.is_empty() {
        return Err("login and password are required".into());
    }
    let jar = Jar(std::env::temp_dir().join(format!("ti_bz_jar_{}.txt", std::process::id())));
    let jar_s = jar.0.to_string_lossy().into_owned();

    eprintln!("logging in ...");
    curl(&format!("url = \"{BASE}/login\"\ndata = \"email={}&password={}\"\n", urlenc(&email), urlenc(&password)), &["-c", &jar_s, "-o", "NUL"], false)?;
    drop(password);
    let page = curl(&format!("url = \"{BASE}/userfiles\"\n"), &["-b", &jar_s, "-L", "-w", "\n%{http_code} %{url_effective}"], false)?;
    let tail = page.lines().last().unwrap_or("");
    if !tail.starts_with("200") || tail.contains("login") {
        return Err(format!("login failed (storage page answered: {tail})"));
    }
    eprintln!("logged in");

    let mut ok = false;
    for attempt in 1..=3 {
        eprintln!("uploading titanium_v2.so ({:.1} MB), attempt {attempt} (about 10 minutes) ...", so.len() as f64 / 1e6);
        let form = format!("{}", so_path.display()).replace('\\', "/");
        let r = curl(
            &format!("url = \"{BASE}/userfile/uploadfile\"\nform = \"file=@{form};type=application/octet-stream\"\nform = \"path=\"\nform = \"filename=titanium_v2.so\"\n"),
            &["-b", &jar_s, "-c", &jar_s, "--max-time", "3000", "-#"],
            true,
        );
        match r {
            Ok(body) if body.replace(' ', "").contains("\"success\":true") => {
                ok = true;
                break;
            }
            Ok(body) => eprintln!("upload answered: {}", body.chars().take(300).collect::<String>()),
            Err(e) => eprintln!("upload attempt failed: {e}"),
        }
    }
    if !ok {
        return Err("the .so upload did not succeed".into());
    }

    eprintln!("verifying: downloading /userfiles/titanium_v2.so back ...");
    let tmp = std::env::temp_dir().join(format!("ti_bz_verify_{}.so", std::process::id()));
    curl(&format!("url = \"{BASE}/userfiles/titanium_v2.so\"\n"), &["-b", &jar_s, "--max-time", "1800", "-o", &tmp.to_string_lossy()], false)?;
    let back = std::fs::read(&tmp).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&tmp);
    if hex(&sha256(&back)) != so_sha {
        return Err(format!("VERIFY FAILED: the downloaded file has sha256 {} (expected {so_sha}); no bot version was created", &hex(&sha256(&back))[..16]));
    }
    eprintln!("verified: storage /titanium_v2.so == local file ({})", &so_sha[..16]);

    if no_version {
        eprintln!("done (library only; no bot version created)");
        return Ok(());
    }
    eprintln!("setting keep-running ...");
    let m = curl(
        &format!("url = \"{BASE}/mybots/modify\"\ndata = \"botid={BOT_ID}&enable_keep_running=true&simpleio=true&opensource=false&description=Titanium+Ataxx%3A+Python+driver+%2B+Rust+engine+%28persistent%29\"\n"),
        &["-b", &jar_s],
        false,
    )?;
    eprintln!("  modify: {}", m.chars().take(200).collect::<String>());
    eprintln!("creating the bot version ...");
    let py_form = py_path.display().to_string().replace('\\', "/");
    let r = curl(
        &format!(
            "url = \"{BASE}/mybots/create\"\nform = \"id={BOT_ID}\"\nform = \"name={BOT_NAME}\"\nform = \"game={GAME_ID}\"\nform = \"extension=py36\"\nform = \"simpleio=true\"\nform = \"enable_keep_running=true\"\nform = \"opensource=false\"\nform = \"description={desc}\"\nform = \"source=@{py_form};type=text/x-python;filename=titanium_botzone.py\"\n"
        ),
        &["-b", &jar_s],
        false,
    )?;
    eprintln!("  create: {}", r.chars().take(300).collect::<String>());
    eprintln!("done. Check the new version on the bot page (and that it is the active one).");
    Ok(())
}
