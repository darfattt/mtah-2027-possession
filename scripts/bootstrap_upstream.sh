#!/usr/bin/env bash
# Clone the official challenge repo into upstream/ (scorer, validator, public GT, baselines).
# Re-run to pull updates. Safe: upstream/ is gitignored.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d upstream/.git ]; then
  git -C upstream pull --ff-only
else
  git clone https://github.com/metrica-sports/more-than-a-hack-2027 upstream
fi

# Let the upstream starter see the same data folder.
if [ ! -e upstream/data ] || [ -L upstream/data ]; then
  rm -rf upstream/data
  ln -s "$(pwd)/data" upstream/data
fi

pip install -r upstream/requirements.txt
echo "upstream ready. Try: python upstream/evaluation/run_evaluation.py"
