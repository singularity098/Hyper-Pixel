# Hyper Pixel — notes for Claude

Desktop scanner app for Linux, written in Rust with a Slint UI. GPL-3.0-or-later.
Owner's scanner: **Canon CanoScan LiDE 400** (SANE `pixma` backend, A4 bed 216×297 mm,
up to 4800 dpi). Defaults are tuned for it.

## Workflow rules
- **Always commit and push directly to `main`** (owner's explicit choice). No PRs unless asked.
- Keep this file current: update it whenever architecture, conventions or milestones change.
- Before pushing: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace --release`.
- **Pure Rust dependencies only** (owner's choice): no C/C++ libraries linked at build time.
  libsane is loaded at runtime with `libloading`. It is never linked or bundled, because the
  host's drivers and `/etc/sane.d` config must be used.
- The photo splitter is the most important feature. Any change to `hp-core/src/detect.rs`
  must keep or improve the benchmark numbers (see below).

## Layout
| Crate | Purpose |
|---|---|
| `crates/hp-core` | All image processing, no GUI or scanner deps. `detect` (photo splitter), `crop` (rotated extraction, keeps 8/16-bit), `color` (presets), `document` (deskew/whiten/Sauvola), `export` (all formats), `pdf` (minimal writer + invisible OCR text layer), `pipeline` (shared extract/name/save), `synth` (synthetic scenes with ground truth), `eval` (scoring), `geom`, `raster`. |
| `crates/hp-scan` | `Backend`/`Device` traits; `sane` (FFI via libloading), `mock` (demo scanners `mock:photos`, `mock:documents`). |
| `crates/hp-ocr` | `ocrs`/`rten` wrapper, model lookup (`$HYPER_PIXEL_MODELS`, `$APPDIR/usr/share/hyper-pixel/models`, XDG data dir) and download. |
| `crates/hp-cli` | `hyper-pixel-cli`: `split`, `ocr`, `eval`, `eval-synth`, `fetch-models`, `devices`. |
| `crates/hp-app` | `hyper-pixel` GUI. `ui/app.slint` (fluent-dark style), `main.rs` (state + wiring, app state in a thread_local `App`), `worker.rs` (scanner thread: SANE must stay on one thread), `editor.rs` (crop-box interaction, unit-tested), `settings.rs` (persisted JSON in `~/.config/hyper-pixel`). |

## Key conventions
- Detections and crop boxes are `RotatedRect`s in **bed millimetres** (`ScanGeometry`
  maps between pixels and mm using dpi and origin). This lets preview boxes map onto
  final scans of any resolution or sub-area.
- Final photo scans only scan the union of the boxes plus 4 mm. Boxes the user has not
  edited are re-fitted on the full-resolution scan (`Editor::refine_with`).
- `RotatedRect` angles are in degrees, clockwise on screen (y points down), and
  normalised to (-45, 45].
- Debug builds compile dependencies with opt-level 3 so image code stays usable.

## Detector benchmark
`cargo run --release -p hp-core --example eval_synth -- <n> <dpi> <dark|light|mix> <gap_mm>`
(with `HP_DUMP=<dir>`, it writes overlays of the failures) or `hyper-pixel-cli eval-synth`.
Baseline at 75 dpi with a 5 mm gap, 200 scenes:
- Dark background: 95% of scenes perfect, mean corner error 0.07 mm. The failures are
  near-black photos on black.
- Light background: 100% of scenes perfect, mean corner error about 0.5 mm. This is a
  bias from the synthetic shadow and is absorbed by the 0.4 mm inset.

Real-scan corpus: put `name.png` and `name.json` (`{"dpi":..,"photos":[RotatedRect mm]}`)
in `samples/`, then run `hyper-pixel-cli eval samples`. `split --write-truth` creates a
starting JSON. Private scans go in `samples/private/` (gitignored).

## Running without hardware
`HYPER_PIXEL_MOCK=1 cargo run --release -p hp-app`. Demo scanners also appear
automatically when SANE finds no devices. For CI, SANE's `test` backend is enabled in
`/etc/sane.d/dll.conf`.

## Packaging
`packaging/build-appimage.sh` builds the AppImage, bundling the OCR models. The
`.github/workflows/ci.yml` workflow runs fmt, clippy, tests and the benchmark, builds the
AppImage, and publishes a `continuous` prerelease on every push to main. Tags `v*`
create releases.

## Roadmap / ideas
- Real-scan corpus from the owner's LiDE 400, then tune the detector on it. Particular
  cases: dark photos on black, white borders on the white lid, touching prints.
- ML restoration (dust/scratch removal, denoise, colourisation) via `rten` models.
- Automatic photo orientation (faces/horizon) to set `quarter_turns`.
- Scanner hardware buttons (SANE sensor options) to trigger scans.
- EXIF date-taken / description editing for archives.
