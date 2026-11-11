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
| `crates/hp-core` | All image processing, no GUI or scanner deps. `detect` (photo splitter), `crop` (rotated extraction, keeps 8/16-bit), `color` (presets, `is_monochrome`), `restore` (dust & scratch removal), `document` (deskew/whiten/Sauvola), `export` (all formats), `pdf` (minimal writer + invisible OCR text layer), `pipeline` (shared extract/enhance/name/save, `Enhancer` hook for neural steps), `synth` (synthetic scenes with ground truth), `synth_dust` (synthetic dusty prints + scoring), `eval` (scoring), `geom`, `raster`. |
| `crates/hp-scan` | `Backend`/`Device` traits; `sane` (FFI via libloading), `mock` (demo scanners `mock:photos`, `mock:documents`). |
| `crates/hp-ocr` | `ocrs`/`rten` wrapper, model lookup (`$HYPER_PIXEL_MODELS`, `$APPDIR/usr/share/hyper-pixel/models`, XDG data dir) and download. |
| `crates/hp-ml` | CPU neural models via `rten`: `inpaint` (LaMa, 512² windows), `colorize` (DDColor), `models` (registry, download with SHA-256 check, streaming zip extraction), `MlEnhancer` (implements `pipeline::Enhancer`). |
| `crates/hp-cli` | `hyper-pixel-cli`: `split`, `restore`, `ocr`, `eval`, `eval-synth`, `fetch-models [ocr\|ai\|all]`, `devices`. |
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
- Corner magnifier: the canvas shows a loupe (`Loupe` in `app.slint`) beside the pointer when it
  is on a box corner, while dragging a corner or drawing a box (`Editor::is_placing_point`), or
  while Shift is held. The mouse wheel sets its zoom (2–16×). Both settings are persisted.
- Custom window frame: `MainWindow` has `no-frame: true`. The toolbar's empty space and the
  sidebar logo header are `DragArea`s (drag moves via winit `drag_window`, double click maximises).
  Minimise/maximise/close are `WindowButton`s at the right of the toolbar (close is red only on
  hover). `ResizeEdge` strips call `drag_resize_window`. This needs slint's `unstable-winit-030`
  feature. After a WM move or resize, `end_grab` sends Slint the release the WM swallowed;
  without it, later clicks all go to the drag area.
- Sidebar padding (18 px) is set per row, not on the outer layout, so the `ScrollView` spans the
  panel and its scrollbar sits in the right gutter instead of over the controls.
- The sidebar logo is the SVG (`assets/hyper-pixel.svg`), rasterised at its displayed size.
  Shrinking the 512 px PNG aliases.
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

## Restoration
Per photo, `pipeline::enhance_photo` runs: dust removal → colour preset → colourisation (only
if `color::is_monochrome`; toned prints count as mono; skipped with the Monochrome preset).

Dust & scratches (`hp-core/src/restore.rs`), all at full resolution, sizes in mm via dpi:
- Background = `close(open(Y))` with a square a bit larger than the biggest speck; bright and
  dark top-hats against that one background (separate open/close gave opposite-polarity
  halos next to every defect).
- Threshold = max(min contrast, k × local texture energy over 0.5 mm). Hysteresis at 60%.
- Each candidate is refined against a plane fitted to its surrounding ring (keeps pixels with
  ≥40% of peak deviation; drops plateaus the square opening leaves in texture).
- **Edge sharpness is the key discriminator**: dust sits on the print surface and is crisp,
  print detail is softened by the print. Median over the boundary of (largest 1-px step /
  change over a 4-px walk across the edge) must exceed `min_sharpness` (0.5 Normal; ×1.15 at
  ~300 dpi; off below 250 dpi, where print detail is as crisp as dust).
- Then size/shape (speck diameter or line mean width), and colour (a defect must not add
  chroma over its surroundings). Accepted hairs/scratches are followed from their ends into
  fainter stretches (≤2 mm, must stay thin).
- Filling: healing brush (texture from the best-matching offset by ring SSD + harmonic
  correction, long defects in chunks); thin (<2.5 px) regions get plain harmonic fill.
  Pixels outside the mask stay bit-identical.
- `RUST_LOG=hp_core::restore=trace` logs each accept/reject with its reason.

AI (`hp-ml`, optional download ~410 MB, both Apache-2.0):
- LaMa (Carve ONNX, fixed 512²): ~3 s per window on 4 cores. The export does **not** blank
  the hole itself (masked pixels must be zeroed) and continues whatever borders the hole, so
  the hole is grown by a pixel. On dust-sized defects it scored *worse* than the healing
  brush (`hp-ml/examples/eval_dust_ml.rs`), so it only redoes defects ≥0.4 mm wide or
  ≥0.4 mm² (`InpaintParams`). It shines on genuinely large holes.
- DDColor (Qualcomm AI Hub ONNX of `ddcolor_paper_tiny`, 256² input, external `.data` file):
  feed the L channel as grey RGB, get CIE Lab a/b, upsample bilinearly, recombine with the
  full-resolution L. ~0.4 s per photo. Keeps 16-bit.
- Model URLs/hashes live in `hp-ml/src/models.rs`. `examples/ml_demo.rs` (`fetch`,
  `inpaint`, `raw`, `colorize`) is the quickest way to try them.

Colour correction: a density-space dye-fade model (`D = k·D₀ + stain` per channel) was tried
and measured worse than the existing Restore faded preset on synthetic fades, so not shipped.

## Dust benchmark
`cargo run --release -p hp-core --example eval_dust -- <n> <dpi> <light|normal|strong>`
(`HP_DUMP`, `HP_MISSED=<dir>` for crops of misses, `HP_TRUTH_MASK=1` to score filling alone,
`HP_VERBOSE=1`). 50×35 mm hostile synthetic prints with 40 defects each. Baseline, 50 scenes:
- 600 dpi Normal: 92.9% of visible defects found, defect-pixel error 69.9 → 5.3 (0..255),
  global error 0.418 → 0.051, ~4 small false regions per patch. ~0.2 s per patch (≈1.8 s for
  a 6×4" print).
- Light: 83.6%, 0.7 false regions per patch. Strong: ~92%, ~5 per patch (meant for larger
  defects than the synthetic set has).
- 300 dpi Normal: 89.1%, ~6 false regions per patch.
Real dusty scans would be the better yardstick; none are in the corpus yet.

## Running without hardware
`HYPER_PIXEL_MOCK=1 cargo run --release -p hp-app`. Demo scanners also appear
automatically when SANE finds no devices. For CI, SANE's `test` backend is enabled in
`/etc/sane.d/dll.conf`.

## Packaging
`packaging/build-appimage.sh` builds the AppImage, bundling the OCR models. The
`.github/workflows/ci.yml` workflow runs fmt, clippy, tests and the benchmark, builds the
AppImage, and publishes a `continuous` prerelease on every push to main. Tags `v*`
create releases.

Icon: `assets/hyper-pixel.svg` is the source (a stack of prints, the front one dissolving into
pixels; no background tile, the viewBox is fitted to the artwork so it fills the icon). `assets/hyper-pixel.png` (512) and `hyper-pixel-128.png` are
rendered from it with headless Chromium, not drawn by hand; re-render them after editing the SVG.
Wordmark: "Hyper Pixel" semibold with the tagline "SCAN · SPLIT · RESTORE" letter-spaced below.

## Roadmap / ideas
- Grow the real-scan corpus (`samples/lide400/` has 3 scans): white-bordered prints on the
  white lid, overlapping prints with no gap, near-black photos on black.
- Real dusty scans in the corpus, to tune `restore` beyond the synthetic set.
- A manual repair brush in the GUI (paint damage, fill with LaMa); a glass-dust map from a
  blank-bed scan (dust on the glass sits at fixed bed positions).
- Larger DDColor variant (needs our own ONNX export; the hosted one is the tiny 256² model).
- Automatic photo orientation (faces/horizon) to set `quarter_turns`.
- Scanner hardware buttons (SANE sensor options) to trigger scans.
- EXIF date-taken / description editing for archives.
