# automodelling

Training-first tabular modelling project structured in the spirit of `karpathy/autoresearch`, but upgraded for more realistic production-style experimentation.

The project is split into:

- `prepare.py` - fixed data loading, feature planning, splits, metrics, and logging harness
- `train.py` - editable training loop where you iterate on models and hyperparameters
- `program.md` - experiment rules and iteration guidance
- `automodelling.py` - top-level entrypoint for running, inspecting, and bootstrapping experiments

## What makes this more real-world

The harness now does more than a plain baseline sweep:

- fixed train/validation split and cross-validation for fair comparisons
- robust numeric preprocessing with quantile clipping, imputation, and scaling
- rare-category-aware one-hot encoding with category grouping
- automatic dropping of constant and almost-entirely-missing columns
- richer diagnostics for high-cardinality and identifier-like columns
- imbalance-aware classification candidates
- PyTorch tabular MLP candidates as part of the same main model search
- binary threshold tuning on the training split
- stronger single-model and ensemble candidates
- persistent experiment registry with keep/discard status
- visualization artifacts for history, candidate comparison, and deep-learning curves
- agent-facing reports that mention what changed in params and config across runs
- dataset-aware search planning that decides whether to emphasize trees, regularized baselines, or deep learning first

This makes iteration closer to an actual modelling workflow: edit `train.py`, re-run one experiment, inspect the registry, and keep only meaningful gains.

The top-level `automodelling.py` script is now more agentic than `train.py`: it can start from a dataset path or URL, infer a sensible initial task when the user gives only the dataset, run a short autonomous search over multiple profiles, and promote the best artifact.

There are two intended entrypoints:

- human-friendly: `python automodelling.py --dataset ... --output ...`
- agent / LLM-friendly: run the same search, then call `python automodelling.py inspect --output ... --json` for pure machine-readable state

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Program JSON

Create a small `program.json` to define the task:

```json
{
  "goal": "Predict customer churn for telecom users",
  "dataset": "/absolute/path/to/customers.csv",
  "target": "churn",
  "test_size": 0.2,
  "cv_folds": 5,
  "random_state": 42,
  "drop_high_missing_threshold": 0.98,
  "categorical_min_frequency": 0.01,
  "categorical_max_categories": 50,
  "numeric_clip_quantile": 0.01,
  "binary_threshold_metric": "balanced_accuracy",
  "enable_deep_learning": true,
  "deep_learning_hidden_dims": [256, 128],
  "deep_learning_dropout": 0.1,
  "deep_learning_learning_rate": 0.001,
  "deep_learning_weight_decay": 0.0001,
  "deep_learning_batch_size": 256,
  "deep_learning_max_epochs": 30,
  "deep_learning_patience": 6,
  "deep_learning_validation_fraction": 0.15,
  "deep_learning_device": "auto"
}
```

The `dataset` field can be a local path or a direct URL. `file://...` sources also work.

Useful config fields:

- `drop_high_missing_threshold` drops feature columns that are mostly empty
- `categorical_min_frequency` groups rare categories during one-hot encoding
- `categorical_max_categories` limits one-hot expansion
- `numeric_clip_quantile` clips extreme numeric tails before scaling
- `binary_threshold_metric` chooses how the classification threshold is tuned on the training split
- `enable_deep_learning` toggles PyTorch tabular candidates on or off
- `deep_learning_*` controls hidden sizes, optimizer settings, early stopping, and device selection
- `candidate_profile` lets you override the search family manually, but the agentic frontend now tries to choose the ordering from the dataset first

## Run the Generic Agentic Search

This is the generic frontend for “give it a dataset and iterate”:

```bash
python automodelling.py --dataset /path/to/data.csv --target churn --output runs/churn_agentic --max-experiments 5
```

You can also point it at a URL:

```bash
python automodelling.py --dataset https://example.com/data.csv --target churn --output runs/churn_agentic --max-experiments 5
```

If `--goal` is omitted, the tool will generate a reasonable default goal from the inferred or explicit target.

The search now starts by understanding the dataset shape and feature mix, then prioritizes the model family order. For example:

- mostly numeric and large/wide datasets push deep learning earlier
- heavy categorical or high-cardinality datasets keep tree-heavy search earlier
- smaller or noisier datasets keep regularized baselines in the loop

## Run a Single Experiment

```bash
python train.py --program program.json --output runs/churn_v1 --description "baseline real-world sweep"
```

The top-level entrypoint also supports explicit single-run mode:

```bash
python automodelling.py run --program program.json --output runs/churn_v1 --description "baseline real-world sweep"
```

You can also use explicit modes:

```bash
python automodelling.py --dataset /path/to/data.csv --output runs/agentic_auto --max-experiments 5
python automodelling.py agent --dataset /path/to/data.csv --output runs/agentic_auto --max-experiments 5
python automodelling.py search --program program.json --output runs/churn_agentic --max-experiments 5
python automodelling.py run --program program.json --output runs/churn_single --description "baseline real-world sweep"
python automodelling.py inspect --output runs/churn_v1
python automodelling.py inspect --output runs/churn_v1 --json
python automodelling.py init-program --path program.json --goal "Predict churn" --dataset /path/to/data.csv --target churn
```

## Default candidate families

Classification:

- logistic regression
- random forest
- extra trees
- histogram gradient boosting
- soft-voting ensemble
- torch tabular MLP when `torch` is available and deep learning is enabled

Regression:

- ridge
- elastic net
- huber regression
- random forest
- extra trees
- histogram gradient boosting
- voting ensemble
- torch tabular MLP when `torch` is available and deep learning is enabled

## Metrics

Primary metrics:

- binary classification: `roc_auc`
- multiclass classification: `f1_weighted`
- regression: `r2`

Secondary metrics:

- classification: `accuracy`, `balanced_accuracy`, `f1_weighted`, `precision_positive`, `recall_positive`, `roc_auc`, `log_loss`, `brier_score`
- regression: `r2`, `mae`, `rmse`, `median_ae`

## Logged artifacts

Each run writes:

- `runs/<name>/dataset_profile.json` - fixed EDA, feature plan, and warnings
- `runs/<name>/results.tsv` - one-line experiment registry
- `runs/<name>/latest_summary.json` - full JSON summary for the latest experiment
- `runs/<name>/experiments/exp_XXXX.json` - detailed per-experiment report
- `runs/<name>/experiments/exp_XXXX_agent_report.md` - agent-oriented summary with score deltas and parameter changes
- `runs/<name>/experiments/exp_XXXX_validation_predictions.csv` - validation predictions for the winning candidate from that experiment
- `runs/<name>/plots/improvement_history.png` - score history across experiments
- `runs/<name>/plots/exp_XXXX_candidate_scores.png` - latest candidate comparison for the tracked metric
- `runs/<name>/plots/exp_XXXX_training_curve.png` - training curve when the winning model is a deep learning model
- `runs/<name>/agentic_search_summary.md` - top-level autonomous search summary when you use `automodelling.py` in search mode
- `runs/<name>/agentic_manifest.json` - machine-readable run manifest for other agents or LLM tools

If the experiment beats the previous best primary score, it also updates:

- `runs/<name>/best_model.joblib`
- `runs/<name>/best_summary.json`
- `runs/<name>/best_validation_predictions.csv`

## How to iterate well

1. Keep `prepare.py` stable.
2. Edit `train.py` to change model families, hyperparameters, or ensembling logic.
3. Re-run with a clear `--description`.
4. Compare `results.tsv`, threshold details, train-vs-validation gap, and warnings before keeping the change.
