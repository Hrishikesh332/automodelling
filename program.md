# AutoModelling Program

This file is the main contract for the coding agent.

If you are using this repository in the `autoresearch` style, the agent should start here, read this file first, and then iterate.

This repository follows an `autoresearch`-style layout for tabular modelling:

- `prepare.py` is the fixed evaluation harness.
- `train.py` is the editable training loop.
- `planning.py` is the search planner with heuristic fallback and optional LLM-guided next-step selection.
- `agent.py` is the top-level repo entrypoint that runs the agentic workflow.
- `results.tsv` is the experiment registry.
- `runs/<name>/experiments/*.json` stores the full metric history for each run.

At the top level, `agent.py` runs the autonomous multi-experiment search, while `automodelling.py` remains the backend runner.

## Iteration rules

1. Read this file first.
2. Keep the split, preprocessing harness, and metrics stable in `prepare.py`.
3. Make modelling changes mainly in `train.py`.
4. Change `planning.py` only if you intentionally want to change the search policy.
5. Record each experiment with a short `--description` so the registry explains what changed.
6. Prefer changes that improve the primary metric without widening the generalization gap.
7. Read the dataset warnings before trusting gains from suspicious features.

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

1. Start the agent with a dataset and output directory:

```bash
python agent.py --dataset /path/to/data.csv --target target_column --output runs/my_task_agentic --max-experiments 5
```

2. If you want a repeatable config, create a program JSON:

```bash
python agent.py init-program --path program.json --dataset /path/to/data.csv --target target_column --goal "Predict target"
```

3. If you want the backend without the repo wrapper, the equivalent command is:

```bash
python automodelling.py agent --dataset /path/to/data.csv --target target_column --output runs/my_task_agentic --max-experiments 5
```

4. If you want an external LLM planner to choose the next experiment:

```bash
python agent.py --dataset /path/to/data.csv --target target_column --output runs/my_task_agentic --max-experiments 5 --search-planner llm --llm-planner-command "python /path/to/planner.py"
```

5. For another agent or an LLM tool, inspect the run with:

```bash
python agent.py inspect --output runs/my_task_agentic --json
```

That path inspects the dataset first and then decides whether to emphasize deep learning, tree-heavy search, or regularized search early in the iteration sequence.

6. Inspect:

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

7. Edit `train.py` and repeat, keeping the harness fixed.
