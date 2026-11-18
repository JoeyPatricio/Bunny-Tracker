#!/usr/bin/env bash
# Fetch ONNX Runtime 1.29.0 headers and the Linux library for this machine
# from Microsoft's NuGet package. Same version as the runtime venv's
# onnxruntime pin. GitHub's release tarballs returned 404 when this was
# written (2026-11-17), and the nupkg carries the same C/C++ API.
#
#   agent-cpp/scripts/fetch-onnxruntime.sh [dest]
#
# dest defaults to ~/.cache/bunnycam/onnxruntime-1.29.0, which is also
# CMake's default ONNXRUNTIME_ROOT. Needs curl and unzip.
set -euo pipefail

VERSION=1.29.0
SHA256=4557021fec4da3924e2e12ff44b1cc50b176de0d2c5c7daa89ab11c208b3a454
URL="https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$VERSION/microsoft.ml.onnxruntime.$VERSION.nupkg"
DEST="${1:-$HOME/.cache/bunnycam/onnxruntime-$VERSION}"

case "$(uname -m)" in
  x86_64) RID=linux-x64 ;;
  aarch64) RID=linux-arm64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

if [ -f "$DEST/lib/libonnxruntime.so" ] && [ -f "$DEST/include/onnxruntime_cxx_api.h" ]; then
  echo "ONNX Runtime $VERSION already at $DEST"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -sSfL -o "$tmp/ort.nupkg" "$URL"
echo "$SHA256  $tmp/ort.nupkg" | sha256sum -c --quiet
unzip -q "$tmp/ort.nupkg" 'build/native/include/*' "runtimes/$RID/native/*" -d "$tmp/x"

mkdir -p "$DEST"
rm -rf "$DEST/include" "$DEST/lib"
mv "$tmp/x/build/native/include" "$DEST/include"
mv "$tmp/x/runtimes/$RID/native" "$DEST/lib"
echo "ONNX Runtime $VERSION ($RID) installed at $DEST"
