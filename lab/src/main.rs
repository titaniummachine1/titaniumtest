//! `lab`: research studies of the champion eval, outside the hub (D957).
//!   lab hce-fit --tdat X.tdat [--train 20000] [--val 2000] [--seed 7] [--out FILE]
//!   lab see-screen --tdat X.tdat [--rows 50000] [--seed 7] [--bench]
mod deepsee;
mod radius;
mod hcecmp;
mod hcefast;
mod hcetrain;
mod extval;
mod hcefit;
mod zattrain;
mod pairs;
mod seescreen;
mod util;
mod hceeval;
mod gxeval;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("hce-fit") => hcefit::run(&args[1..]),
        Some("see-screen") => seescreen::run(&args[1..]),
        Some("deep-see") => deepsee::run(&args[1..]),
        Some("radius") => radius::run(&args[1..]),
        Some("hce-fast") => hcefast::run(&args[1..]),
        Some("hce-compare") => hcecmp::run(&args[1..]),
        Some("hce-train") => hcetrain::run(&args[1..]),
        Some("val-fen") => extval::fen(&args[1..]),
        Some("val-score") => extval::score(&args[1..]),
        Some("zataxx-val") => extval::zataxx(&args[1..]),
        Some("zataxx-train") => zattrain::run(&args[1..]),
        Some("hce-eval") => hceeval::cli(&args[1..]),
        Some("gx-eval") => gxeval::cli(&args[1..]),
        _ => {
            eprintln!("usage: lab hce-fit ... | lab see-screen ...");
            ExitCode::from(2)
        }
    }
}
