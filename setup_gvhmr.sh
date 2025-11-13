#!/usr/bin/env bash
set -euo pipefail

#===============================================================================
# GVHMR 環境セットアップ（uv + venv / PyTorch nightly cu128 / CUDA 12.8+）
# - 事前: CUDA Toolkit 12.8+ を導入し、nvcc が使えること（WSL では apt 推奨）
# - 目的: Blackwell(sm_120) GPU で GVHMR を動かす
#===============================================================================

#=== config ====================================================================
PY_VER="3.10"
UV_PRIMARY_INDEX="https://download.pytorch.org/whl/nightly/cu128"  # torch/vision nightly (CUDA 12.8+)
UV_EXTRA_INDEX="https://pypi.org/simple"
VENV_DIR=".venv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAR_UV_BUILD_CACHE="${CLEAR_UV_BUILD_CACHE:-0}"  # 1 にすると uv ビルドキャッシュを掃除して再同期
#===============================================================================

cd "$REPO_ROOT"
echo ">> Working directory: $PWD"

# 0) システムツール（必要なら） ---------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  echo ">> Installing system build tools (may require sudo)..."
  if [ "${EUID:-$UID}" -ne 0 ]; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config git curl ca-certificates \
      libgl1 ffmpeg
  else
    apt-get update
    apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config git curl ca-certificates \
      libgl1 ffmpeg
  fi
fi

# 1) uv インストール（未導入なら） -------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo ">> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"  # uv の既定インストール先をパスに追加
fi
echo ">> uv version: $(uv --version || true)"

# 2) Python 用意 & venv 作成 ------------------------------------------------------
echo ">> Installing Python ${PY_VER} via uv..."
uv python install "${PY_VER}"

echo ">> Creating venv at ${VENV_DIR}..."
uv venv --python "${PY_VER}" "${VENV_DIR}"

# 3) venv 有効化 ------------------------------------------------------------------
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -V

# 4) /mnt 下のハードリンク警告抑止（WSL）& 複数インデックス横断 ---------------------
export UV_LINK_MODE=copy
export UV_INDEX_STRATEGY=unsafe-best-match

# 5) CUDA の場所を決める（無ければ候補から探索）-----------------------------------
: "${CUDA_HOME:=/usr/local/cuda-12.8}"
if [ ! -d "$CUDA_HOME" ]; then
  for c in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda; do
    if [ -d "$c" ]; then CUDA_HOME="$c"; break; fi
  done
fi
echo ">> CUDA_HOME: $CUDA_HOME"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# 5.1) nvcc の存在チェック（早期に停止）
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Install CUDA Toolkit (12.8+) and set CUDA_HOME." >&2
  echo "       e.g., sudo apt-get install cuda-toolkit-12-8 (or 12-9) on WSL Ubuntu." >&2
  exit 1
fi
nvcc --version

# 5.2) Blackwell (sm_120) を確実に含める
export TORCH_CUDA_ARCH_LIST="12.0"

# 6) lock & sync（PyTorch nightly CUDA12.8 を第一インデックスに）-------------------
echo ">> uv lock ..."
uv lock \
  --index-url "${UV_PRIMARY_INDEX}" \
  --extra-index-url "${UV_EXTRA_INDEX}" \
  --preview-features extra-build-dependencies

if [ "${CLEAR_UV_BUILD_CACHE}" = "1" ]; then
  echo ">> Clearing uv build cache (~/.cache/uv/builds-v0)"
  rm -rf ~/.cache/uv/builds-v0 || true
fi

echo ">> uv sync ..."
uv sync \
  --index-url "${UV_PRIMARY_INDEX}" \
  --extra-index-url "${UV_EXTRA_INDEX}" \
  --preview-features extra-build-dependencies

# 7) プロジェクト編集可能インストール ----------------------------------------------
echo ">> Editable install of the project (-e .)"
uv pip install -e .

# 8) I/O ディレクトリ作成 ---------------------------------------------------------
mkdir -p inputs outputs inputs/checkpoints inputs/checkpoints/body_models/{smpl,smplx}

# 9) 動作確認 ---------------------------------------------------------------------
python - <<'PY'
import torch, sys
print("torch:", getattr(torch, "__version__", "n/a"), "cuda:", getattr(torch, "version", type("x",(object,),{})()).cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    try:
        print("arch list:", torch.cuda.get_arch_list())
    except Exception as e:
        print("arch list: <unavailable>", e)
print("python:", sys.version.split()[0])
PY

echo ">> DONE."
echo "   - Activate env: source ${VENV_DIR}/bin/activate"
echo "   - Example run : uv run python tools/demo/demo.py --video=/path/to/your.mp4 -s"
echo "   - Persist CUDA vars (recommended): add CUDA_HOME/PATH/LD_LIBRARY_PATH to ~/.bashrc or /etc/profile.d/cuda.sh"
