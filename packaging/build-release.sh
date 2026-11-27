#!/usr/bin/env bash
# Build dist/HyperScan-<version>-linux-<arch>.tar.gz: the whole app (GUI and command line) as
# one binary, `hyperscan`, with the OCR models embedded.
#
# libsane is deliberately NOT bundled: the host's SANE library, drivers (backends) and
# /etc/sane.d configuration must be used, and the app loads libsane at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${VERSION:-$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)}"
ARCH="$(uname -m)"
MODELS="$(pwd)/dist/ocr-models"

mkdir -p "$MODELS"
for m in text-detection.onnx text-recognition.onnx; do
    [ -f "$MODELS/$m" ] && continue
    if [ -f "$HOME/.local/share/hyperscan/models/$m" ]; then
        cp "$HOME/.local/share/hyperscan/models/$m" "$MODELS/$m"
    else
        curl -fsSL --retry 3 -o "$MODELS/$m.part" "https://ocrs-models.s3-accelerate.amazonaws.com/$m"
        mv "$MODELS/$m.part" "$MODELS/$m"
    fi
done

HYPERSCAN_BUNDLE_OCR="$MODELS" cargo build --release --locked -p hp-app

# The archive holds just the binary, named as it is typed: hyperscan.
OUT="dist/HyperScan-${VERSION}-linux-${ARCH}.tar.gz"
STAGE="target/release-package"
rm -rf "$STAGE" && mkdir -p "$STAGE"
install -m755 target/release/hyperscan "$STAGE/hyperscan"
tar -C "$STAGE" -czf "$OUT" hyperscan
(cd dist && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
echo "Built $OUT"
