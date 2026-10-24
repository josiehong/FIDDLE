#!/usr/bin/env bash
# Install SIRIUS into ./tools/ for the comparison experiments; compare.sh puts
# it on the PATH. Run from the repository root. SIRIUS_VERSION overrides the
# pinned version.
set -e

SIRIUS_VERSION=${SIRIUS_VERSION:-6.5.4}

case "$(uname -s)" in
    Linux) os=linux; bin=tools/sirius/bin ;;
    Darwin) os=macos; bin=tools/sirius.app/Contents/MacOS ;;
    *) echo "No SIRIUS build for $(uname -s)"; exit 1 ;;
esac
case "$(uname -m)" in
    x86_64 | amd64) arch=x64 ;;
    arm64 | aarch64) arch=arm64 ;;
    *) echo "No SIRIUS build for $(uname -m)"; exit 1 ;;
esac

zip=sirius-$SIRIUS_VERSION-$os-$arch.zip
url=https://github.com/sirius-ms/sirius/releases/download/v$SIRIUS_VERSION/$zip
mkdir -p tools
curl -fL -o tools/$zip $url
curl -fL -o tools/$zip.sha256 $url.sha256
if command -v sha256sum >/dev/null; then
    (cd tools && sha256sum -c $zip.sha256)
else
    (cd tools && shasum -a 256 -c $zip.sha256)
fi
rm -rf tools/sirius tools/sirius.app
unzip -q tools/$zip -d tools
rm tools/$zip tools/$zip.sha256

$bin/sirius --version
echo "Installed SIRIUS $SIRIUS_VERSION ($bin/sirius)."
echo "SIRIUS needs a free account (https://account.bright-giant.com). Log in once with:"
echo "  $bin/sirius login -u <email> -p"
