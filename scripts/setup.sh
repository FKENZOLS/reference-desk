#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REQUESTED="${1:-auto}"
PYTHON="${PYTHON:-python3}"

case "$REQUESTED" in auto|cuda|rocm|cpu) ;; *) echo "Use: $0 [auto|cuda|rocm|cpu]"; exit 2 ;; esac

BACKEND="$REQUESTED"
if [[ "$BACKEND" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    BACKEND="cuda"
  elif command -v rocminfo >/dev/null 2>&1 || [[ -e /dev/kfd ]]; then
    BACKEND="rocm"
  else
    BACKEND="cpu"
  fi
fi

echo "Preparing Reference Desk for $BACKEND"
[[ -x .venv/bin/python ]] || "$PYTHON" -m venv .venv
VPY="$ROOT/.venv/bin/python"
"$VPY" -m pip install --upgrade pip wheel
case "$BACKEND" in
  cuda) "$VPY" -m pip install -r requirements-cuda.txt ;;
  rocm) "$VPY" -m pip install -r requirements-rocm-linux.txt ;;
  cpu) "$VPY" -m pip install -r requirements-cpu.txt ;;
esac
"$VPY" -m pip install -r requirements-base.txt
printf '%s\n' "$BACKEND" > .rag-profile
"$VPY" scripts/cache_models.py

if command -v ollama >/dev/null 2>&1; then
  ollama pull qwen3-embedding:0.6b
else
  echo "Ollama was not found. Install it, then run: ollama pull qwen3-embedding:0.6b"
fi

"$VPY" scripts/doctor.py --expect "$BACKEND"
echo "Ready. Start with ./start.sh"
