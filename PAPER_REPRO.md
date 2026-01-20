# Paper Repro Guide

This repository supports reproducible runs for baseline (event-triggered, 2 actions) and SPH (3 actions) settings. The core entry points are `train_ppo.py`, `experiments/eval_suite.py`, and `scripts/make_paper_artifacts.py`.

## Environment setup

```bash
conda create -n empc python=3.8 -y
conda activate empc
pip install -r requirements.txt
```

## Quick smoke tests (Windows paths)

Baseline:

```bash
python train_ppo.py --episodes 5 --seed 0 --rho 0 --run_dir results\smoke_baseline\run_test --sph 0
```

SPH:

```bash
python train_ppo.py --episodes 5 --seed 0 --rho 0 --run_dir results\smoke_sph\run_test --sph 1 --np_eco 10 --np_pro 30 --path_set single --curriculum 0
```

Evaluation suite:

```bash
python experiments\eval_suite.py --run_dir results\smoke_baseline\run_test --episodes 3 --seed 0 --path_set single
```

Paper artifacts (assumes results under `results\paper_runs`):

```bash
python scripts\make_paper_artifacts.py --results_root results\paper_runs --out_dir paper_artifacts
```

## Full experiment grid (Windows PowerShell)

```powershell
.\scripts\run_paper_grid.ps1
```

To override the default episode count (500):

```powershell
.\scripts\run_paper_grid.ps1 -Episodes 100
```

## Configuration switches

`train_ppo.py` supports:
- `--sph {0,1}`: baseline (2 actions) or SPH (3 actions)
- `--np_eco`, `--np_pro`: horizons for SPH actions
- `--path_set {single,multi}`: single Dang path or multi-path mix
- `--curriculum {0,1}`: curriculum over path sets
- `--domain_rand {0,1}`: domain randomization of initial state + path params
- `--eval_suite {0,1}`: trigger fixed evaluation suite every evaluation interval
- `--difficulty`: scaling for randomization strength

Output structure per run:
- `train_logs.csv`: per-eval summaries
- `summary.json`: latest evaluation summary (including solve stats)
- `checkpoints/ppo_best.pt`: best checkpoint by evaluation return
- `results/*.mat`: detailed rollout data

