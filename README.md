# The Execution Reality Gap

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce results

```bash
make partA   # Part A — slippage waterfall (results/partA_*)
make partB   # Part B — follower analysis + optimisations (results/partB_*)
make partC   # Part C — gate policy comparison (results/partC_*)
make all     # everything
```

All outputs (CSV tables, PNG figures) are written to `results/`.
Deterministic under the fixed random seed.
