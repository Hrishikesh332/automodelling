# AutoModelling Program

This repository follows an `autoresearch`-style layout for tabular modelling:

- `prepare.py` is the fixed evaluation harness.
- `train.py` is the editable training loop.
- `planning.py` is the search planner with heuristic fallback and optional LLM-guided next-step selection.
- `results.tsv` is the experiment registry.
- `runs/<name>/experiments/*.json` stores the full metric history for each run.

At the top level, `automodelling.py` can now run an autonomous multi-experiment search from a dataset path or URL.

## Iteration rules

1. Keep the split, preprocessing harness, and metrics stable in `prepare.py`.
2. Make modelling changes in `train.py`: candidate models, hyperparameters, weighting, or ensembling.
3. Record each experiment with a short `--description` so the registry explains what changed.
4. Prefer changes that improve the primary metric without widening the generalization gap.
5. Read the dataset warnings before trusting gains from suspicious features.

## What the harness does automatically

- drops constant columns and almost-fully-missing columns
- groups rare categories during one-hot encoding
- clips numeric outliers before scaling
- monitors high-cardinality and identifier-like columns
- checks duplicate rows, target leakage signals, and split drift
- tunes a binary decision threshold on the training split
- can include PyTorch tabular MLP candidates for deep learning runs
- logs both cross-validation and holdout validation performance
- writes agent reports and visualization artifacts for experiment history and candidate comparisons
- orders the autonomous search using dataset understanding before trying search profiles
- writes production handoff artifacts such as a model card, feature schema, and prediction contract
- writes ablation artifacts so isolated improvements can be separated from bundled changes

## Primary metrics

- binary classification: `roc_auc`
- multiclass classification: `f1_weighted`
- regression: `r2`

## Secondary metrics to monitor

- classification: `accuracy`, `balanced_accuracy`, `f1_weighted`, `precision_positive`, `recall_positive`, `roc_auc`, `log_loss`, `brier_score`
- regression: `r2`, `mae`, `rmse`, `median_ae`

## Recommended workflow

1. Create a program JSON with the modelling goal, dataset path, and target.
2. Run a baseline sweep:

```bash
python train.py --program program.json --output runs/my_task --description "baseline real-world sweep"
```

Or let the agentic frontend iterate automatically:

```bash
python automodelling.py --dataset /path/to/data.csv --target target_column --output runs/my_task_agentic --max-experiments 5
```

Or let an external LLM planner choose the next experiments:

```bash
python automodelling.py --dataset /path/to/data.csv --target target_column --output runs/my_task_agentic --max-experiments 5 --search-planner llm --llm-planner-command "python /path/to/planner.py"
```

For another agent or an LLM tool, inspect the run with:

```bash
python automodelling.py inspect --output runs/my_task_agentic --json
```

That path now inspects the dataset first and then decides whether to emphasize deep learning, tree-heavy search, or regularized search early in the iteration sequence.

3. Inspect:

- `runs/my_task/results.tsv`
- `runs/my_task/latest_summary.json`
- `runs/my_task/experiments/exp_XXXX.json`
- `runs/my_task/experiments/exp_XXXX_agent_report.md`
- `runs/my_task/best_validation_predictions.csv`
- `runs/my_task/plots/improvement_history.png`
- `runs/my_task/agentic_search_summary.md`
- `runs/my_task/handoff/latest_model_card.md`
- `runs/my_task/handoff/prediction_contract.json`
- `runs/my_task/analysis/ablation_summary.md`
- `runs/my_task/analysis/ablation_table.tsv`

4. Edit `train.py` and repeat, keeping the harness fixed.
