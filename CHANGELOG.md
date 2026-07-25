# Changelog

All notable changes to VUGVA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-25

### Added

- GPU discovery and topology detection (`gpu.rs`)
- Virtual Memory Table with 5-state page machine (`vmt.rs`)
- V1 allocator with fast-path/single-GPU optimization (`allocator.rs`)
- V2 tiered allocator with three-tier hierarchy (VRAM→DRAM→SSD) (`tiered.rs`)
- 64-byte DMA descriptor ring (`dma.rs`)
- NUMA-aware routing with distance-based bandwidth factors (`gpu.rs`)
- Look-Ahead Attention Tracking prefetch engine (`prefetch.rs`)
- CUDA stream/event management (`streams.rs`)
- NVRTC runtime kernel compilation (`nvrtc_kernel.rs`)
- dlopen(3) based FFI loader — zero external dependencies (`ffi/`)
- CUDA 12 compatibility (`cuCtxCreate_v2`)
- Linker fix (`#[link(name = "cuda")]`)
- Working demo binary (`examples/demo.rs`)
- 23 unit tests
- 19 hardware integration tests
- Paper verification: all 8 invariants validated on RTX 4060
- Landing page with Three.js 3D effects and glassmorphism UI
- Documentation: Getting Started, Architecture, Benchmarks, API Reference
- Paper (PDF) compiled from LaTeX source
- AGPL-3.0 license
- GitHub community files (SECURITY, CONTRIBUTING, CODE_OF_CONDUCT, issue templates)
- GitHub Actions CI (check, fmt, clippy, tests, build)
