# Real scans from the CanoScan LiDE 400

Ground truth for three 600 dpi full-bed scans of four prints each. The scans themselves are
private family photos and are not in the repository; ask the owner for them and put them
here as `<name>.png` next to the JSON files, then run:

```bash
cargo run --release -p hp-cli -- eval samples/lide400
cargo run --release -p hp-cli -- eval --at-dpi 150 samples/lide400   # as a preview would see it
```

| File | Background | What makes it hard |
|---|---|---|
| `black-foam-overlap` | black foam | foam grain, dark photos on dark foam, a wedge-shaped 1–3.5 mm gap between the top pair, the bottom pair 1.5 mm apart |
| `white-lid-touching` | white lid | top pair butted together (only the white paper edge between them), a sky photo whose light top barely differs from the lid, a photo running off the right edge, lid-edge shadow along the top and left border |
| `white-lid-touching-edges` | white lid | three photos running off the bed edges (most of the scan border is photo), top pair butted together |

The truth was fitted at full resolution from edge profiles and checked corner by corner by
eye; edges are accurate to about 0.05 mm, apart from a ±0.25 mm waviness of the cut edges
themselves. Where a photo runs off the scan, the truth is the visible part.
