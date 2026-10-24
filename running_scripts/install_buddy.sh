#!/usr/bin/env bash
# Install BUDDY (msbuddy) into the active Python environment for the comparison
# experiments. msbuddy's models are LightGBM, which it does not declare as a
# dependency. MSBUDDY_VERSION overrides the pinned version.
set -e

MSBUDDY_VERSION=${MSBUDDY_VERSION:-0.3.16}

pip install "msbuddy==$MSBUDDY_VERSION" lightgbm scikit-learn
msbuddy --help >/dev/null
python -c "import lightgbm, sklearn"
echo "Installed msbuddy $MSBUDDY_VERSION."
