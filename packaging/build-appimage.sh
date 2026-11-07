#!/usr/bin/env bash
# Build dist/Hyper_Pixel-<version>-x86_64.AppImage.
#
# libsane is deliberately NOT bundled: the host's SANE library, drivers (backends) and
# /etc/sane.d configuration must be used, and the app loads libsane at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${VERSION:-$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)}"
ARCH="${ARCH:-x86_64}"
APPDIR="$(pwd)/dist/AppDir"
TOOLS="$(pwd)/dist/tools"

cargo build --release --locked -p hp-app -p hp-cli

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/512x512/apps" "$APPDIR/usr/share/icons/hicolor/scalable/apps" \
         "$APPDIR/usr/share/hyper-pixel/models" "$TOOLS"
install -m755 target/release/hyper-pixel target/release/hyper-pixel-cli "$APPDIR/usr/bin/"
install -m644 packaging/hyper-pixel.desktop "$APPDIR/usr/share/applications/"
install -m644 assets/hyper-pixel.png "$APPDIR/usr/share/icons/hicolor/512x512/apps/hyper-pixel.png"
install -m644 assets/hyper-pixel.svg "$APPDIR/usr/share/icons/hicolor/scalable/apps/hyper-pixel.svg"
cp packaging/hyper-pixel.desktop assets/hyper-pixel.png "$APPDIR/"
ln -sf hyper-pixel.png "$APPDIR/.DirIcon"
install -m755 packaging/AppRun "$APPDIR/AppRun"

# Bundle OCR models so OCR works offline out of the box.
for m in text-detection.onnx text-recognition.onnx; do
    dest="$APPDIR/usr/share/hyper-pixel/models/$m"
    if [ -f "$HOME/.local/share/hyper-pixel/models/$m" ]; then
        cp "$HOME/.local/share/hyper-pixel/models/$m" "$dest"
    else
        curl -fsSL --retry 3 -o "$dest" "https://ocrs-models.s3-accelerate.amazonaws.com/$m"
    fi
done

if [ ! -x "$TOOLS/appimagetool" ]; then
    curl -fsSL --retry 3 -o "$TOOLS/appimagetool" \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage"
    chmod +x "$TOOLS/appimagetool"
fi

OUT="dist/Hyper_Pixel-${VERSION}-${ARCH}.AppImage"
APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$ARCH" "$TOOLS/appimagetool" --no-appstream "$APPDIR" "$OUT"
echo "Built $OUT"
