# Hyper Pixel — notes for Claude

Desktop scanner app for Linux, written in Rust. GPL-3.0-or-later.

## Workflow rules
- **Always commit and push directly to `main`** (owner's explicit choice). No PRs unless asked.
- Keep this file current: update it whenever architecture, conventions or milestones change.
- Before pushing: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
- Pure Rust dependencies preferred (no C/C++ libs linked at build time). libsane is loaded at
  runtime with `libloading`, never linked or bundled.
