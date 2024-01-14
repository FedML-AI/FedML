#!/usr/bin/env bash
set -e

if ninja --version &>/dev/null; then
  echo "ninja version $(ninja --version) is correctly installed. Installing flash attention"

  pip install flash-attn --no-build-isolation
else
  echo "ninja is either not installed or corrupted."
  echo "reinstall ninja via \"pip uninstall -y ninja && pip install ninja\""
  echo "see https://github.com/Dao-AILab/flash-attention for detail"
fi
