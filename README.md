# Neural surrogates for the MetaRL look-ahead indicators (GBWM)

Replaces the expensive look-ahead simulation that produces `g_sim` and `p_sim` in the MetaRL
goals-based-wealth-management policy with three small MLPs, keeping the trained PPO actors unchanged.
Paper: `paper/paper.pdf` (source `paper/paper.tex`).

## Setup
1. Python 3.10 and `pip install -r requirements.txt`.
2. Put the advisor's **minimal-metarl** package (not included here) in `external/minimal-metarl/`
   (it must contain `run_rl_simulation.py`, `infl_test_rl_functions.py`, `models/`, ...), or set
   `METARL_DIR=/path/to/minimal-metarl`.
3. Run every command from the repository root. The first call of the exact simulation compiles with
   Numba (about 90 s per new process); set `NUMBA_CACHE_DIR=./nbcache` to reuse it.

## Pipeline (leakage-free study, `src/`)
| Step | Command | Output |
|---|---|---|
| Sanity: pre-clip features reproduce the env state | `python src/verify_reconstruction.py` | max abs diff 0.0 |
| Sanity: my MC loop equals the released `RLMC_pathbased` | `python src/validate_teacher.py` | rewards match |
| Build dataset (all 66 cases) | `python src/build_dataset_c1.py 500 24 data/dataset_c1.npz` | `data/dataset_c1.npz` (not committed) |
| Train, hold out Case 1 group | `python src/train_surrogates.py` | `data/surrogates_c1.npz` |
| Train 3 group-held-out folds | `for f in 0 1 2; do python src/train_cv_fold.py data/dataset_c1.npz data/sur_fold$f.npz $f; done` | `data/sur_fold{0,1,2}.npz` |
| Case 1 time-by-wealth grid | `python src/eval_case.py 1` | `out/eval_case1.csv` |
| Policy figures | `python src/plot_policy_heatmaps.py --csv out/eval_case1.csv --out figs/case1 --paper` | figures |
| Monte Carlo utility, 66 cases | `python src/mc_eval.py out/mc.json 5000 0.58` (last arg: wealth ratio; optional case list `1-66`) | per-case utility |
| Summary | `python src/summarize_mc.py out/mc.json` | |
| Full-MC compute time | `python src/mc_time.py out/time.json 10000 0.58` | |
| Diagnose a case | `python src/diag.py 26` | step-level teacher vs surrogate |

`src/compare_strategies.py` is the earlier strategy-comparison script.
Trained surrogate weights are in `data/` so the evaluation scripts run without retraining.

## Development study (`dev_study/`)
`build_dataset.py`, `gen_states.py`, `exp1_surrogate.py`, `exp2_timing_ablation.py` produced the 5-fold CV and
latency tables. They import the advisor's package, so copy them into the `minimal-metarl` folder (or add it to
`PYTHONPATH`) and run from there. Their features are taken *after* the environment's filter/clip, so accuracy
may be optimistic (see paper, Section 7).

## Results (`results/`)
Monte Carlo JSON for wealth ratios 0.25 / 0.58 / 3 / 10 and the 10,000-path timing run; Case 1 grid evaluation.

## Paper
`cd paper && python make_latency_fig.py ../dev_study/exp2_results.json figs/latency && python make_mc_fig.py && pdflatex paper && pdflatex paper`

## Caveats
Utility is measured relative to the original MetaRL, not DP (the minimal package has no DP solver). All results use one actor seed.
The next-best goal option is never a take, so the student funds a goal only when the actor follows a best option that is a take.
