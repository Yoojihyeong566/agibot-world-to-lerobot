#!/usr/bin/env bash
# Create the `agibot_world` conda env and apply the two non-obvious fixes that
# AgiBot World + lerobot need on Linux. Idempotent — safe to re-run.
#
# Usage:  bash scripts/setup_env.sh [ENV_NAME]
set -euo pipefail

ENV_NAME="${1:-agibot_world}"

# --- locate conda --------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
  done
fi
command -v conda >/dev/null 2>&1 || { echo "conda not found"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- create env from environment.yml ------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "env '${ENV_NAME}' already exists — updating"
  conda env update -n "${ENV_NAME}" -f "${HERE}/environment.yml"
else
  conda env create -n "${ENV_NAME}" -f "${HERE}/environment.yml"
fi
conda activate "${ENV_NAME}"

# --- install this package (agibot2lerobot) ------------------------------------
pip install -e "${HERE}"

# --- FIX 1: torchcodec needs ffmpeg 4..7 (NOT 8) ------------------------------
# environment.yml already pins ffmpeg=7.*; assert it here.
FFV="$(ffmpeg -version 2>/dev/null | head -1 | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' || echo '?')"
if [ "${FFV}" -gt 7 ] 2>/dev/null; then
  echo "downgrading ffmpeg ${FFV} -> 7 (torchcodec compatibility)"
  conda install -y -c conda-forge "ffmpeg=7.*"
fi

# --- FIX 2: make conda's newer libstdc++ load first (CXXABI_1.3.15) ------------
# ffmpeg pulls libopenvino which needs CXXABI_1.3.15; the system libstdc++ may
# be too old. Prepend $CONDA_PREFIX/lib on env activation.
HOOK_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${HOOK_DIR}"
cat > "${HOOK_DIR}/zz_ld_library_path.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# --- verify --------------------------------------------------------------------
echo "--- verifying torchcodec ---"
python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')"
echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
