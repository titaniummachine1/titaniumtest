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
mod nodecost;
mod boundcmp;
mod evalgate;
mod groupdiff;
mod spairs;
mod win;
mod archloc;
mod netscale;
mod standard;
mod stopctl;
mod jat3;
mod jat3x;
mod featbench;
mod tfbench;
mod levatt;
mod wsprobe;
mod blendspsa;
mod siblingcorr;
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
        Some("jat3-train") => jat3::run(&args[1..]),
        Some("jat3-bench") => jat3::bench(&args[1..]),
        Some("jat3-compare") => jat3::compare(&args[1..]),
        Some("jat3-explain") => jat3x::explain(&args[1..]),
        Some("feat-bench") => featbench::run(&args[1..]),
        Some("tf-bench") => tfbench::run(&args[1..]),
        Some("levatt-train") => levatt::run(&args[1..]),
        Some("ws-probe") => wsprobe::run(&args[1..]),
        Some("loss-elo-corr") => blendspsa::run_corr(&args[1..]),
        Some("spsa-blend") => blendspsa::run_spsa(&args[1..]),
        Some("sibling-corr") => siblingcorr::run(&args[1..]),
        Some("hce-eval") => hceeval::cli(&args[1..]),
        Some("node-cost") => nodecost::run(&args[1..]),
        Some("incr-stats") => nodecost::incr_stats(&args[1..]),
        Some("weak-cost") => nodecost::weak_cost(&args[1..]),
        Some("bound-cmp") => boundcmp::run(&args[1..]),
        Some("hce-ablate") => hceeval::ablate(&args[1..]),
        Some("gx-eval") => gxeval::cli(&args[1..]),
        Some("eval-gate") => evalgate::run(&args[1..]),
        Some("group-diff") => groupdiff::run(&args[1..]),
        Some("arch-locality") => archloc::run(&args[1..]),
        Some("residual-check") => hcetrain::residual_check(&args[1..]),
        Some("residual-pair") => hcetrain::residual_pair(&args[1..]),
        Some("net-scale") => netscale::run(&args[1..]),
        _ => {
            eprintln!("usage: lab hce-fit ... | lab see-screen ...");
            ExitCode::from(2)
        }
    }
}
