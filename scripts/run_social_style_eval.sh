#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m app.evals.social_style_eval "$@"
