//! pxl — a fast batch image processor.
//!
//! The library exposes the resampling kernels ([`kernel`]), the per-file
//! operations ([`ops`]) and the parallel batch driver ([`pipeline`]); the
//! `pxl` binary is a thin CLI over [`pipeline`].

pub mod kernel;
pub mod ops;
pub mod pipeline;
