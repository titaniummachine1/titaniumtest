//! Opening books: load + check. A gate never runs on an unchecked book (2026-10-30: book50k.txt was ~70%
//! lines from the old one-stone-per-side start and every gate on it measured mostly the wrong game).

use titanium::Board;

/// Squares of the a7 and g1 2x2 corners (the black start corners of the standard position).
const BLACK_START_CORNERS: [usize; 8] = [0, 1, 7, 8, 40, 41, 47, 48];
/// Lines from the old one-stone start show both black start corners empty this early.
const SIGNATURE_MAX_STONES: usize = 16;
/// Refuse a book when more than this share of lines carries the one-stone-start signature.
pub const MAX_BAD_SHARE: f64 = 0.01;

#[derive(Debug, PartialEq)]
pub struct BookReport {
    pub lines: usize,
    pub one_stone_signature: usize,
    pub duplicates: usize,
}

/// Check raw book text. Err = refuse (malformed line, duplicates, or too many wrong-start lines).
pub fn check(text: &str) -> Result<BookReport, String> {
    let mut seen = std::collections::HashSet::new();
    let mut rep = BookReport { lines: 0, one_stone_signature: 0, duplicates: 0 };
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let board = line.split_whitespace().next().unwrap_or("");
        if board.len() != 49 || Board::from_str(line).is_none() {
            return Err(format!("book line {}: malformed {:?}", n + 1, &line[..line.len().min(60)]));
        }
        rep.lines += 1;
        if !seen.insert(line.to_string()) {
            rep.duplicates += 1;
        }
        let cells = board.as_bytes();
        let stones = cells.iter().filter(|&&c| c == b'x' || c == b'o').count();
        let corners_empty = BLACK_START_CORNERS.iter().all(|&i| cells[i] != b'x' && cells[i] != b'o');
        if stones <= SIGNATURE_MAX_STONES && corners_empty {
            rep.one_stone_signature += 1;
        }
    }
    if rep.lines == 0 {
        return Err("book is empty".into());
    }
    if rep.duplicates > 0 {
        return Err(format!("book has {} duplicate lines", rep.duplicates));
    }
    if rep.one_stone_signature as f64 > MAX_BAD_SHARE * rep.lines as f64 {
        return Err(format!(
            "{}/{} lines look like the old one-stone start (a7 and g1 corners empty at <= {} stones)",
            rep.one_stone_signature, rep.lines, SIGNATURE_MAX_STONES
        ));
    }
    Ok(rep)
}

/// Load a checked book: one board per line (49 cells `.xo#` + side `b`/`w`).
pub fn load(path: &str) -> Result<Vec<Board>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read book {path}: {e}"))?;
    check(&text).map_err(|e| format!("book {path} refused: {e}"))?;
    Ok(text.lines().map(str::trim).filter(|l| !l.is_empty()).map(|l| Board::from_str(l).unwrap()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(board: &str, side: &str) -> String {
        format!("{board} {side}\n")
    }

    fn standard_variant(i: usize) -> String {
        let mut c: Vec<u8> = Board::start().to_string().split_whitespace().next().unwrap().bytes().collect();
        c[i] = b'x';
        String::from_utf8(c).unwrap()
    }

    #[test]
    fn standard_lines_pass() {
        let text: String = [8, 9, 10, 15, 16].iter().map(|&i| line(&standard_variant(i), "w")).collect();
        assert_eq!(check(&text).unwrap().lines, 5);
    }

    #[test]
    fn one_stone_start_lines_refused() {
        // old bug start: black a1 (sq 42), white g7 (sq 6), plus one black clone
        let text: String = [35, 36, 43].iter().map(|&i| {
            let mut c = vec![b'.'; 49];
            c[42] = b'x';
            c[6] = b'o';
            c[i] = b'x';
            line(&String::from_utf8(c).unwrap(), "w")
        }).collect();
        assert!(check(&text).unwrap_err().contains("one-stone start"));
    }

    #[test]
    fn duplicates_and_malformed_refused() {
        let l = line(&standard_variant(8), "w");
        assert!(check(&(l.clone() + &l)).unwrap_err().contains("duplicate"));
        assert!(check("xo b\n").unwrap_err().contains("malformed"));
        assert!(check("").is_err());
    }
}
