#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

accelerate launch trainer.py -opt configs/train.yml "$@"
