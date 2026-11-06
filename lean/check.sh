#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -d ../.elan/toolchains ]]; then
  export ELAN_HOME="$(cd ../.elan && pwd)"
fi
lake build
