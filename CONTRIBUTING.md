# Contributing to VUGVA

Thank you for your interest in contributing to VUGVA! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Open a new issue using the **Bug Report** template
3. Include:
   - GPU model and compute capability
   - CUDA version and driver version
   - OS and kernel version
   - Minimal reproduction steps
   - Expected vs actual behavior

### Suggesting Features

1. Open a new issue using the **Feature Request** template
2. Explain the use case and motivation
3. Reference relevant paper sections if applicable

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Ensure all tests pass: `cargo test`
5. Ensure clippy is clean: `cargo clippy --all-targets`
6. Ensure formatting is correct: `cargo fmt`
7. Commit with a clear message
8. Open a Pull Request

## Development Setup

### Prerequisites

- Rust 1.70+ (stable)
- NVIDIA GPU with compute capability sm_60+
- libcuda.so (CUDA driver library)
- libnvrtc.so (optional, for runtime compilation)

### Building

```bash
cargo build
```

### Running Tests

```bash
# Unit tests only (no GPU required)
cargo test --lib

# Hardware integration tests (requires GPU)
cargo test --test hardware

# All tests
cargo test
```

### Code Style

- Follow existing code conventions
- Use `cargo fmt` before committing
- Use `cargo clippy --all-targets` and fix all warnings
- Add tests for new functionality
- Keep changes minimal and focused

## Architecture

```
vugva/src/
├── lib.rs          # Crate root, error types, constants
├── ffi/
│   ├── mod.rs      # dlopen/dlsym loader
│   ├── cuda.rs     # CUDA FFI declarations
│   └── nvrtc.rs    # NVRTC FFI declarations
├── gpu.rs          # GPU discovery, P2P, NUMA
├── vmt.rs          # Virtual Memory Table
├── allocator.rs    # V1 allocator (Algorithm 1)
├── tiered.rs       # V2 tiered allocator (Algorithm 2)
├── dma.rs          # DMA descriptor ring
├── streams.rs      # CUDA stream/event management
├── prefetch.rs     # Look-Ahead Attention Tracking
└── nvrtc_kernel.rs # NVRTC runtime compilation
```

## Testing Guidelines

- Unit tests should not require GPU hardware
- Hardware tests should gracefully skip if no GPU is available
- Test all paper invariants where applicable
- Test both valid and invalid state transitions
- Test single and multi-GPU configurations

## Pull Request Guidelines

- Keep PRs focused on a single change
- Reference related issues
- Include test results (unit + hardware)
- Update documentation if needed
- Follow existing code style

## License

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](LICENSE).
