//! D1024: library face of the lab, so another engine can call our trained HCE (hceeval::HceEval).
//! The `lab` binary (main.rs) keeps its own module tree; this crate only re-compiles the modules it needs.
#![allow(dead_code)]
mod hcecmp;
mod hcefit;
mod pairs;
mod spairs;
mod util;
mod zattrain;
mod extval;
mod hcetrain;
mod seescreen;
mod deepsee;
mod radius;
mod hcefast;
pub mod hceeval;
pub mod gxeval;
