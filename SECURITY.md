# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within VUGVA, please send an email to irfan@furylogic.com. All security vulnerabilities will be promptly addressed.

**Please do NOT report security vulnerabilities through public GitHub issues.**

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 1 week
- **Fix or mitigation**: within 30 days

### Scope

This policy applies to the VUGVA library code only (Rust source in `vugva/src/`). It does not apply to:

- Third-party CUDA drivers or libraries
- Hardware vulnerabilities
- Issues in downstream projects using VUGVA

## Security Best Practices

When using VUGVA in production:

1. **Never run with `LD_PRELOAD` in untrusted environments** — the CUDA interception layer can be exploited if the preload path is writable by untrusted users.
2. **Validate GPU ordinals** — only pass trusted GPU ordinals to `VugvaEngine::new()`.
3. **Monitor VRAM usage** — OOM conditions on GPU can cause undefined behavior in CUDA drivers.
4. **Keep CUDA drivers updated** — VUGVA loads `libcuda.so` at runtime; vulnerabilities in the driver affect all users.
