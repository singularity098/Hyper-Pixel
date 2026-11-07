# Hyper Pixel — notes for Claude

Desktop scanner app for Linux, written in Rust with a Slint UI. GPL-3.0-or-later.
Owner's scanner: **Canon CanoScan LiDE 400** (SANE `pixma` backend, A4 bed 216×297 mm,
up to 4800 dpi). Defaults are tuned for it.

## Workflow rules
- **Always commit and push directly to `main`** (owner's explicit choice). No PRs unless asked.
- Keep this file current: update it whenever architecture, conventions or milestones change.
- Toolchain is pinned in `rust-toolchain.toml`; bump it deliberately (and fix new lints).
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
  edited are re-fitted on the full-resolution scan (`editor::refine_boxes`). If the final
  scan finds several photos inside one untouched box, the box is split into them.
- `RotatedRect` angles are in degrees, clockwise on screen (y points down), and
  normalised to (-45, 45].
- Debug builds compile dependencies with opt-level 3 so image code stays usable.
- Preview default is 150 dpi (thin seams between touching prints vanish at 75 dpi). A saved
  75 from older versions is migrated on load.

## Detector (`hp-core/src/detect.rs`)
Two resolutions: a 100 dpi *working* level for masks and components, and a *fine* level
(300 dpi, capped at the scan dpi) for seams and edges. Key ideas, each fixing a real-scan
failure:
- Background = the ring colour mode (found in gamma RGB, where noise is even) that appears
  most widely around the border. Photos running off the bed can cover most of the border.
- A 1.2 mm band along the scan border is replaced by its inside neighbour (`replicate_border`;
  0.8 mm at the fine level): the LiDE shows frame strips and a lid-edge shadow there.
- Seam splitting (`best_seam`): a Radon-style search for straight lines across a blob with
  gap, paper-edge-ridge or edge evidence, then re-centred on the gap by RANSAC. Each half
  keeps the seam as a `Limit` that refinement may not cross.
- Edge refinement takes the strongest rise of background distance that has background just
  outside it. Two passes (±3 mm, then ±1 mm). Coarse extents come from projection
  histograms (ignoring appendages) and from the min-area rect, and the larger wins unless
  its edges are clearly less supported.
- `HP_DETECT_DEBUG=<dir>` dumps masks and seam evidence; `RUST_LOG=hp_core=trace` logs the
  decisions.

## Detector benchmark
Synthetic: `cargo run --release -p hp-core --example eval_synth -- <n> <dpi> <dark|light|mix> <gap_mm>`
(with `HP_DUMP=<dir>`, it writes overlays of the failures; `HP_SCENES=<dir>` saves every
scene with its truth, for `hyper-pixel-cli eval`) or `hyper-pixel-cli eval-synth`.
Baseline at 75 dpi with a 5 mm gap, 200 scenes:
- Dark background: 190/200 scenes perfect, mean corner error 0.09 mm. The failures are
  near-black photos on black.
- Light background: 200/200 perfect, mean corner error 0.26 mm (a bias from the synthetic
  shadow, absorbed by the 0.4 mm inset).
- Touching photos (`mix`, gap 0): 191/200 perfect, mean 0.26 mm.

Real scans (the main yardstick): `samples/lide400/` holds ground truth for three of the
owner's LiDE 400 scans (white lid and black foam, touching prints, photos running off the
bed). The PNGs are private and gitignored; see the README there. `hyper-pixel-cli eval
samples/lide400` (add `--at-dpi 150` to simulate a preview). Baseline: at 600 and 300 dpi,
12/12 photos, mean worst-corner error 0.18 mm, max 0.70 mm. At 150 dpi, 12/12, mean
0.50 mm. At 75 dpi, the white-lid top pairs merge, which is why the preview default is
150 dpi.

New corpus scans: put `name.png` and `name.json` (`{"dpi":..,"photos":[RotatedRect mm]}`)
in a `samples/` folder and run `hyper-pixel-cli eval <folder>`. `split --write-truth` creates
a starting JSON (it never overwrites one). Private scans go in `samples/private/`
(gitignored).

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
- Grow the real-scan corpus (`samples/lide400/` has 3 scans): white-bordered prints on the
  white lid, overlapping prints with no gap, near-black photos on black.
- ML restoration (dust/scratch removal, denoise, colourisation) via `rten` models.
- Automatic photo orientation (faces/horizon) to set `quarter_turns`.
- Scanner hardware buttons (SANE sensor options) to trigger scans.
- EXIF date-taken / description editing for archives.
