
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge

from deep_learning import TorchTabularClassifier, TorchTabularRegressor, isTorchAvailable
from prepare import (
    ExperimentConfig,
    IMPROVEMENT_EPSILON,
    PreparedExperiment,
    appendResultsRow,
    configToDict,
    currentBestScore,
    evaluateCandidate,
    formatMetric,
    loadProgram,
    nextExperimentId,
    prepareExperiment,
    primaryScore,
    safeJsonDump,
    saveBestModel,
    saveValidationPredictions,
    serializableCandidate,
)
from reporting import (
    buildAblationMetadata,
    diffDicts,
    generateVisualizations,
    loadJsonIfPresent,
    writeAblationArtifacts,
    writeAgentReport,
    writeProductionArtifacts,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    estimator: Any
    notes: str


def profileSettings(profile: str) -> dict[str, Any]:
    settings = {
        "balanced": {
            "tree_estimators": 700,
            "tree_leaf": 2,
            "hist_lr": 0.04,
            "hist_iter": 450,
            "hist_leaf_nodes": 31,
            "ridge_alpha": 1.0,
            "elastic_alpha": 0.001,
            "elastic_l1_ratio": 0.1,
            "dl_hidden_dims": None,
            "dl_dropout": None,
            "dl_epochs_delta": 0,
        },
        "tree_heavy": {
            "tree_estimators": 1100,
            "tree_leaf": 1,
            "hist_lr": 0.03,
            "hist_iter": 600,
            "hist_leaf_nodes": 63,
            "ridge_alpha": 1.0,
            "elastic_alpha": 0.001,
            "elastic_l1_ratio": 0.1,
            "dl_hidden_dims": None,
            "dl_dropout": None,
            "dl_epochs_delta": 0,
        },
        "regularized": {
            "tree_estimators": 500,
            "tree_leaf": 4,
            "hist_lr": 0.025,
            "hist_iter": 350,
            "hist_leaf_nodes": 31,
            "ridge_alpha": 3.0,
            "elastic_alpha": 0.003,
            "elastic_l1_ratio": 0.2,
            "dl_hidden_dims": None,
            "dl_dropout": 0.2,
            "dl_epochs_delta": 0,
        },
        "deep_focus": {
            "tree_estimators": 600,
            "tree_leaf": 2,
            "hist_lr": 0.04,
            "hist_iter": 400,
            "hist_leaf_nodes": 31,
            "ridge_alpha": 1.0,
            "elastic_alpha": 0.001,
            "elastic_l1_ratio": 0.1,
            "dl_hidden_dims": (384, 192, 96),
            "dl_dropout": 0.15,
            "dl_epochs_delta": 8,
        },
        "compact": {
            "tree_estimators": 350,
            "tree_leaf": 3,
            "hist_lr": 0.05,
            "hist_iter": 250,
            "hist_leaf_nodes": 31,
            "ridge_alpha": 1.5,
            "elastic_alpha": 0.002,
            "elastic_l1_ratio": 0.15,
            "dl_hidden_dims": (128, 64),
            "dl_dropout": 0.05,
            "dl_epochs_delta": -10,
        },
    }
    return settings.get(profile, settings["balanced"])


def buildDeepLearningClassifier(prepared: PreparedExperiment, class_weight: str | None) -> CandidateSpec:
    cfg = prepared.config
    profile = profileSettings(cfg.candidate_profile)
    return CandidateSpec(
        "torch_tabular_mlp",
        TorchTabularClassifier(
            hidden_dims=tuple(profile["dl_hidden_dims"] or cfg.deep_learning_hidden_dims),
            dropout=profile["dl_dropout"] if profile["dl_dropout"] is not None else cfg.deep_learning_dropout,
            learning_rate=cfg.deep_learning_learning_rate,
            weight_decay=cfg.deep_learning_weight_decay,
            batch_size=cfg.deep_learning_batch_size,
            max_epochs=max(4, cfg.deep_learning_max_epochs + int(profile["dl_epochs_delta"])),
            patience=cfg.deep_learning_patience,
            validation_fraction=cfg.deep_learning_validation_fraction,
            random_state=cfg.random_state,
            device=cfg.deep_learning_device,
            class_weight=class_weight,
        ),
        "PyTorch MLP for dense tabular features with early stopping and optional class balancing.",
    )


def buildDeepLearningRegressor(prepared: PreparedExperiment) -> CandidateSpec:
    cfg = prepared.config
    profile = profileSettings(cfg.candidate_profile)
    return CandidateSpec(
        "torch_tabular_mlp",
        TorchTabularRegressor(
            hidden_dims=tuple(profile["dl_hidden_dims"] or cfg.deep_learning_hidden_dims),
            dropout=profile["dl_dropout"] if profile["dl_dropout"] is not None else cfg.deep_learning_dropout,
            learning_rate=cfg.deep_learning_learning_rate,
            weight_decay=cfg.deep_learning_weight_decay,
            batch_size=cfg.deep_learning_batch_size,
            max_epochs=max(4, cfg.deep_learning_max_epochs + int(profile["dl_epochs_delta"])),
            patience=cfg.deep_learning_patience,
            validation_fraction=cfg.deep_learning_validation_fraction,
            random_state=cfg.random_state,
            device=cfg.deep_learning_device,
        ),
        "PyTorch MLP regressor for dense tabular representations with early stopping.",
    )


def classificationImbalanceRatio(prepared: PreparedExperiment) -> float:
    target_summary = prepared.dataset_profile.get("target_summary", {})
    return float(target_summary.get("class_imbalance_ratio", 1.0))


def buildClassificationCandidates(prepared: PreparedExperiment) -> list[CandidateSpec]:
    seed = prepared.config.random_state
    profile = profileSettings(prepared.config.candidate_profile)
    imbalance_ratio = classificationImbalanceRatio(prepared)
    use_balanced_weights = imbalance_ratio >= 1.5
    class_weight = "balanced" if use_balanced_weights else None
    tree_weight = "balanced_subsample" if use_balanced_weights else None

    logistic = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight=class_weight,
    )
    random_forest = RandomForestClassifier(
        n_estimators=profile["tree_estimators"],
        min_samples_leaf=profile["tree_leaf"],
        max_features="sqrt",
        class_weight=tree_weight,
        random_state=seed,
        n_jobs=-1,
    )
    extra_trees = ExtraTreesClassifier(
        n_estimators=max(profile["tree_estimators"] + 200, profile["tree_estimators"]),
        min_samples_leaf=profile["tree_leaf"],
        max_features="sqrt",
        class_weight=tree_weight,
        random_state=seed,
        n_jobs=-1,
    )
    hist_gradient = HistGradientBoostingClassifier(
        learning_rate=profile["hist_lr"],
        max_iter=profile["hist_iter"],
        max_leaf_nodes=profile["hist_leaf_nodes"],
        l2_regularization=0.05,
        random_state=seed,
    )

    candidates = [
        CandidateSpec(
            "logistic_regression",
            logistic,
            "Regularized linear baseline with optional class balancing for skewed labels.",
        ),
        CandidateSpec(
            "random_forest",
            random_forest,
            "Bagged tree ensemble with stronger depth control for mixed-type tabular data.",
        ),
        CandidateSpec(
            "extra_trees",
            extra_trees,
            "High-randomization tree ensemble that often works well on messy real-world tables.",
        ),
        CandidateSpec(
            "hist_gradient_boosting",
            hist_gradient,
            "Boosted trees on the engineered dense feature space.",
        ),
        CandidateSpec(
            "soft_voting_ensemble",
            VotingClassifier(
                estimators=[
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=5000,
                            solver="lbfgs",
                            class_weight=class_weight,
                        ),
                    ),
                    (
                        "et",
                        ExtraTreesClassifier(
                            n_estimators=max(300, profile["tree_estimators"] - 100),
                            min_samples_leaf=profile["tree_leaf"],
                            max_features="sqrt",
                            class_weight=tree_weight,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                    (
                        "hgb",
                        HistGradientBoostingClassifier(
                            learning_rate=profile["hist_lr"],
                            max_iter=max(200, profile["hist_iter"] - 100),
                            max_leaf_nodes=profile["hist_leaf_nodes"],
                            l2_regularization=0.05,
                            random_state=seed,
                        ),
                    ),
                ],
                voting="soft",
            ),
            "Probability ensemble that is usually more robust than any single classifier.",
        ),
    ]

    if prepared.config.enable_deep_learning and isTorchAvailable():
        candidates.append(buildDeepLearningClassifier(prepared, class_weight))

    return candidates


def buildRegressionCandidates(prepared: PreparedExperiment) -> list[CandidateSpec]:
    seed = prepared.config.random_state
    profile = profileSettings(prepared.config.candidate_profile)
    ridge = Ridge(alpha=profile["ridge_alpha"])
    elastic = ElasticNet(alpha=profile["elastic_alpha"], l1_ratio=profile["elastic_l1_ratio"], max_iter=5000)
    huber = HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=500)
    random_forest = RandomForestRegressor(
        n_estimators=profile["tree_estimators"],
        min_samples_leaf=profile["tree_leaf"],
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    extra_trees = ExtraTreesRegressor(
        n_estimators=max(profile["tree_estimators"] + 200, profile["tree_estimators"]),
        min_samples_leaf=profile["tree_leaf"],
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    hist_gradient = HistGradientBoostingRegressor(
        learning_rate=profile["hist_lr"],
        max_iter=profile["hist_iter"],
        max_leaf_nodes=profile["hist_leaf_nodes"],
        l2_regularization=0.05,
        random_state=seed,
    )

    candidates = [
        CandidateSpec(
            "ridge",
            ridge,
            "Regularized linear baseline after clipping, imputation, scaling, and one-hot encoding.",
        ),
        CandidateSpec(
            "elastic_net",
            elastic,
            "Sparse linear model for wide feature spaces and mild feature selection.",
        ),
        CandidateSpec(
            "huber_regression",
            huber,
            "Robust linear model that is less sensitive to heavy-tailed noise and outliers.",
        ),
        CandidateSpec(
            "random_forest",
            random_forest,
            "Bagged tree ensemble for non-linear tabular regression structure.",
        ),
        CandidateSpec(
            "extra_trees",
            extra_trees,
            "Aggressive randomization often helps on heterogeneous feature interactions.",
        ),
        CandidateSpec(
            "hist_gradient_boosting",
            hist_gradient,
            "Boosted trees on the engineered dense feature space.",
        ),
        CandidateSpec(
            "voting_ensemble",
            VotingRegressor(
                estimators=[
                    ("ridge", Ridge(alpha=1.0)),
                    (
                        "et",
                        ExtraTreesRegressor(
                            n_estimators=max(300, profile["tree_estimators"] - 100),
                            min_samples_leaf=profile["tree_leaf"],
                            max_features="sqrt",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                    (
                        "hgb",
                        HistGradientBoostingRegressor(
                            learning_rate=profile["hist_lr"],
                            max_iter=max(200, profile["hist_iter"] - 100),
                            max_leaf_nodes=profile["hist_leaf_nodes"],
                            l2_regularization=0.05,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "Blended regressor that stabilizes variance across linear and tree-based inductive biases.",
        ),
    ]

    if prepared.config.enable_deep_learning and isTorchAvailable():
        candidates.append(buildDeepLearningRegressor(prepared))

    return candidates


def buildCandidates(prepared: PreparedExperiment) -> list[CandidateSpec]:
    if prepared.problem_type == "classification":
        return buildClassificationCandidates(prepared)
    return buildRegressionCandidates(prepared)


def rankingKey(prepared: PreparedExperiment, result: dict[str, Any]) -> tuple[float, float, float, float]:
    validation_primary = primaryScore(result, prepared.primary_metric)
    cv_primary = result["cv_summary"]["mean"].get(prepared.primary_metric)
    gap = result.get("generalization_gap")
    fit_seconds = result["fit_seconds"]
    return (
        float("-inf") if validation_primary is None else validation_primary,
        float("-inf") if cv_primary is None else cv_primary,
        float("-inf") if gap is None else -abs(gap),
        -fit_seconds,
    )


def resolveConfig(args: argparse.Namespace) -> ExperimentConfig:
    program_config = loadProgram(args.program)
    if program_config is None:
        if args.dataset is None:
            raise SystemExit("Either --program or --dataset must be provided.")
        config = ExperimentConfig(goal=args.goal or "", dataset=str(args.dataset))
    else:
        config = program_config

    if args.goal is not None:
        config.goal = args.goal
    if args.dataset is not None:
        config.dataset = str(args.dataset)
    if args.target is not None:
        config.target = args.target
    if args.problem_type is not None:
        config.problem_type = args.problem_type
    if args.test_size is not None:
        config.test_size = args.test_size
    if args.cv_folds is not None:
        config.cv_folds = args.cv_folds
    if args.random_state is not None:
        config.random_state = args.random_state
    if getattr(args, "enable_deep_learning", None) is not None:
        config.enable_deep_learning = args.enable_deep_learning
    if getattr(args, "deep_learning_hidden_dims", None):
        config.deep_learning_hidden_dims = tuple(args.deep_learning_hidden_dims)
    if getattr(args, "deep_learning_dropout", None) is not None:
        config.deep_learning_dropout = args.deep_learning_dropout
    if getattr(args, "deep_learning_learning_rate", None) is not None:
        config.deep_learning_learning_rate = args.deep_learning_learning_rate
    if getattr(args, "deep_learning_weight_decay", None) is not None:
        config.deep_learning_weight_decay = args.deep_learning_weight_decay
    if getattr(args, "deep_learning_batch_size", None) is not None:
        config.deep_learning_batch_size = args.deep_learning_batch_size
    if getattr(args, "deep_learning_max_epochs", None) is not None:
        config.deep_learning_max_epochs = args.deep_learning_max_epochs
    if getattr(args, "deep_learning_patience", None) is not None:
        config.deep_learning_patience = args.deep_learning_patience
    if getattr(args, "deep_learning_validation_fraction", None) is not None:
        config.deep_learning_validation_fraction = args.deep_learning_validation_fraction
    if getattr(args, "deep_learning_device", None) is not None:
        config.deep_learning_device = args.deep_learning_device
    if getattr(args, "candidate_profile", None) is not None:
        config.candidate_profile = args.candidate_profile
    return config


def runExperiment(
    prepared: PreparedExperiment,
    description: str,
    plannerContext: dict[str, Any] | None = None,
    plannedChanges: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = perf_counter()
    candidates = buildCandidates(prepared)
    previous_best_summary = loadJsonIfPresent(prepared.output_dir / "best_summary.json")
    previous_latest_summary = loadJsonIfPresent(prepared.output_dir / "latest_summary.json")

    candidate_results: list[dict[str, Any]] = []
    best_spec: CandidateSpec | None = None
    best_result: dict[str, Any] | None = None

    for candidate in candidates:
        result = evaluateCandidate(prepared, candidate.estimator)
        candidate_results.append({"spec": candidate, "result": result})
        if best_result is None or rankingKey(prepared, result) > rankingKey(prepared, best_result):
            best_spec = candidate
            best_result = result

    if best_spec is None or best_result is None:
        raise RuntimeError("No candidate models were evaluated.")

    total_seconds = perf_counter() - start
    results_path = prepared.output_dir / "results.tsv"
    experiment_id = nextExperimentId(results_path)
    previous_best = currentBestScore(results_path, prepared.primary_metric)
    best_primary = primaryScore(best_result, prepared.primary_metric)
    improved = previous_best is None or (
        best_primary is not None and best_primary > previous_best + IMPROVEMENT_EPSILON
    )
    status = "keep" if improved else "discard"
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    best_validation = best_result["validation_metrics"]
    decision_threshold = best_result.get("decision_threshold") or {}
    row = {
        "created_at_utc": created_at_utc,
        "experiment_id": experiment_id,
        "status": status,
        "description": description,
        "problem_type": prepared.problem_type,
        "primary_metric": prepared.primary_metric,
        "primary_score": best_primary,
        "cv_primary_mean": best_result["cv_summary"]["mean"].get(prepared.primary_metric),
        "cv_primary_std": best_result["cv_summary"]["std"].get(prepared.primary_metric),
        "train_primary_score": best_result["train_metrics"].get(prepared.primary_metric),
        "generalization_gap": best_result.get("generalization_gap"),
        "best_model": best_spec.name,
        "fit_seconds": best_result["fit_seconds"],
        "cv_fit_seconds": best_result["cv_fit_seconds"],
        "total_seconds": total_seconds,
        "train_rows": len(prepared.X_train),
        "validation_rows": len(prepared.X_val),
        "train_feature_count": best_result.get("train_feature_count"),
        "decision_threshold": decision_threshold.get("threshold"),
        "threshold_metric": decision_threshold.get("metric"),
        "threshold_score": decision_threshold.get("score"),
        "accuracy": best_validation.get("accuracy"),
        "balanced_accuracy": best_validation.get("balanced_accuracy"),
        "f1_weighted": best_validation.get("f1_weighted"),
        "precision_positive": best_validation.get("precision_positive"),
        "recall_positive": best_validation.get("recall_positive"),
        "roc_auc": best_validation.get("roc_auc"),
        "log_loss": best_validation.get("log_loss"),
        "brier_score": best_validation.get("brier_score"),
        "r2": best_validation.get("r2"),
        "mae": best_validation.get("mae"),
        "rmse": best_validation.get("rmse"),
        "median_ae": best_validation.get("median_ae"),
    }
    appendResultsRow(results_path, row)

    experiments_dir = prepared.output_dir / "experiments"
    validation_path = experiments_dir / f"{experiment_id}_validation_predictions.csv"
    saveValidationPredictions(
        prepared,
        best_result["model"],
        validation_path,
        decisionThreshold=best_result.get("decision_threshold"),
    )

    summary = {
        "created_at_utc": created_at_utc,
        "experiment_id": experiment_id,
        "description": description,
        "status": status,
        "improved_over_previous": improved,
        "previous_best_primary_score": previous_best,
        "goal": prepared.config.goal,
        "dataset_source": prepared.dataset_source,
        "dataset": str(prepared.dataset_path),
        "target_column": prepared.target_column,
        "problem_type": prepared.problem_type,
        "primary_metric": prepared.primary_metric,
        "dataset_profile": prepared.dataset_profile,
        "config": configToDict(prepared.config),
        "best_candidate": {
            "name": best_spec.name,
            "notes": best_spec.notes,
            "estimator_class": best_result["estimator_class"],
            "estimator_params": best_result["estimator_params"],
            "validation_metrics": best_result["validation_metrics"],
            "train_metrics": best_result["train_metrics"],
            "cv_summary": best_result["cv_summary"],
            "fit_seconds": best_result["fit_seconds"],
            "cv_fit_seconds": best_result["cv_fit_seconds"],
            "generalization_gap": best_result["generalization_gap"],
            "decision_threshold": best_result["decision_threshold"],
            "train_feature_count": best_result["train_feature_count"],
            "training_history": best_result.get("training_history"),
            "validation_predictions_path": str(validation_path),
        },
        "candidates": [
            serializableCandidate(item["spec"].name, item["spec"].notes, item["result"])
            for item in candidate_results
        ],
        "artifacts": {
            "results_tsv": str(results_path),
            "dataset_profile": str(prepared.output_dir / "dataset_profile.json"),
            "validation_predictions": str(validation_path),
        },
    }
    if plannerContext is not None:
        summary["planner"] = plannerContext
    summary["planned_changes"] = plannedChanges or {}

    previous_best_candidate = previous_best_summary.get("best_candidate") if previous_best_summary else None
    previous_latest_candidate = previous_latest_summary.get("best_candidate") if previous_latest_summary else None
    score_delta_vs_previous_best = (
        None
        if previous_best_candidate is None or previous_best is None or best_primary is None
        else float(best_primary - previous_best)
    )
    if score_delta_vs_previous_best is not None and abs(score_delta_vs_previous_best) <= IMPROVEMENT_EPSILON:
        score_delta_vs_previous_best = 0.0
    score_delta_vs_previous_latest = (
        None
        if previous_latest_candidate is None
        else float(
            best_primary
            - previous_latest_candidate["validation_metrics"].get(summary["primary_metric"], 0.0)
        )
    )
    if score_delta_vs_previous_latest is not None and abs(score_delta_vs_previous_latest) <= IMPROVEMENT_EPSILON:
        score_delta_vs_previous_latest = 0.0
    summary["comparison"] = {
        "score_delta_vs_previous_best": score_delta_vs_previous_best,
        "score_delta_vs_previous_latest": score_delta_vs_previous_latest,
        "parameter_changes_vs_previous_best": diffDicts(
            previous_best_candidate.get("estimator_params") if previous_best_candidate else None,
            best_result["estimator_params"],
        ),
        "parameter_changes_vs_previous_latest": diffDicts(
            previous_latest_candidate.get("estimator_params") if previous_latest_candidate else None,
            best_result["estimator_params"],
        ),
        "config_changes_vs_previous_best": diffDicts(
            previous_best_summary.get("config") if previous_best_summary else None,
            configToDict(prepared.config),
        ),
        "config_changes_vs_previous_latest": diffDicts(
            previous_latest_summary.get("config") if previous_latest_summary else None,
            configToDict(prepared.config),
        ),
        "previous_best": {
            "experiment_id": previous_best_summary.get("experiment_id") if previous_best_summary else None,
            "best_model": previous_best_candidate.get("name") if previous_best_candidate else None,
        },
        "previous_latest": {
            "experiment_id": previous_latest_summary.get("experiment_id") if previous_latest_summary else None,
            "best_model": previous_latest_candidate.get("name") if previous_latest_candidate else None,
        },
    }
    summary["ablation"] = buildAblationMetadata(summary, IMPROVEMENT_EPSILON)

    if improved:
        best_model_path = prepared.output_dir / "best_model.joblib"
        best_summary_path = prepared.output_dir / "best_summary.json"
        best_predictions_path = prepared.output_dir / "best_validation_predictions.csv"
        saveBestModel(best_result["model"], best_model_path)
        summary["artifacts"]["best_model"] = str(best_model_path)
        summary["artifacts"]["best_summary"] = str(best_summary_path)
        summary["artifacts"]["best_validation_predictions"] = str(best_predictions_path)
        saveValidationPredictions(
            prepared,
            best_result["model"],
            best_predictions_path,
            decisionThreshold=best_result.get("decision_threshold"),
        )

    summary_path = experiments_dir / f"{experiment_id}.json"
    summary["artifacts"]["experiment_summary"] = str(summary_path)
    summary["artifacts"]["latest_summary"] = str(prepared.output_dir / "latest_summary.json")
    visual_artifacts = generateVisualizations(prepared.output_dir, summary)
    summary["artifacts"].update(visual_artifacts)
    productionArtifacts = writeProductionArtifacts(prepared.output_dir, summary)
    summary["artifacts"].update(productionArtifacts)
    agent_report_path = experiments_dir / f"{experiment_id}_agent_report.md"
    summary["artifacts"]["agent_report"] = str(agent_report_path)
    safeJsonDump(summary, summary_path)
    safeJsonDump(summary, prepared.output_dir / "latest_summary.json")
    if improved:
        safeJsonDump(summary, prepared.output_dir / "best_summary.json")

    ablationArtifacts = writeAblationArtifacts(prepared.output_dir)
    summary["artifacts"].update(ablationArtifacts)
    safeJsonDump(summary, summary_path)
    safeJsonDump(summary, prepared.output_dir / "latest_summary.json")
    if improved:
        safeJsonDump(summary, prepared.output_dir / "best_summary.json")
    writeAgentReport(agent_report_path, summary)

    return summary


def printSummary(summary: dict[str, Any]) -> None:
    best = summary["best_candidate"]
    print(f"Experiment: {summary['experiment_id']} ({summary['status']})")
    print(f"Problem type: {summary['problem_type']}")
    print(f"Primary metric: {summary['primary_metric']}")
    print(f"Primary score: {formatMetric(best['validation_metrics'].get(summary['primary_metric']))}")
    print(f"Best model: {best['name']}")
    print(f"Train feature count: {best['train_feature_count']}")
    print(f"CV mean ({summary['primary_metric']}): {formatMetric(best['cv_summary']['mean'].get(summary['primary_metric']))}")
    print(f"CV std ({summary['primary_metric']}): {formatMetric(best['cv_summary']['std'].get(summary['primary_metric']))}")
    if best.get("decision_threshold"):
        threshold = best["decision_threshold"]
        print(
            "Decision threshold: "
            f"{formatMetric(threshold.get('threshold'))} "
            f"({threshold.get('metric')}={formatMetric(threshold.get('score'))})"
        )
    for metric_name, metric_value in best["validation_metrics"].items():
        if metric_name == summary["primary_metric"]:
            continue
        print(f"{metric_name}: {formatMetric(metric_value)}")
    dataset_warnings = summary["dataset_profile"].get("warnings", [])
    if dataset_warnings:
        print("Warnings:")
        for warning in dataset_warnings:
            print(f"- {warning}")
    comparison = summary.get("comparison", {})
    if comparison.get("score_delta_vs_previous_best") is not None:
        print(
            "Delta vs previous best: "
            f"{formatMetric(comparison['score_delta_vs_previous_best'])}"
        )
    param_changes = comparison.get("parameter_changes_vs_previous_best", [])
    if param_changes:
        print(f"Changed params vs previous best: {len(param_changes)}")
    print(f"Validation predictions: {best['validation_predictions_path']}")
    print(f"Experiment summary: {summary['artifacts']['experiment_summary']}")
    if summary["artifacts"].get("agent_report"):
        print(f"Agent report: {summary['artifacts']['agent_report']}")
    if summary["artifacts"].get("history_plot"):
        print(f"History plot: {summary['artifacts']['history_plot']}")
    if summary["artifacts"].get("candidate_plot"):
        print(f"Candidate plot: {summary['artifacts']['candidate_plot']}")
    if summary["artifacts"].get("training_curve_plot"):
        print(f"Training curve plot: {summary['artifacts']['training_curve_plot']}")
    if summary["artifacts"].get("model_card"):
        print(f"Model card: {summary['artifacts']['model_card']}")
    if summary["artifacts"].get("prediction_contract"):
        print(f"Prediction contract: {summary['artifacts']['prediction_contract']}")
    if summary["artifacts"].get("ablation_summary"):
        print(f"Ablation summary: {summary['artifacts']['ablation_summary']}")
    print(f"Results registry: {summary['artifacts']['results_tsv']}")


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and log a single AutoModelling experiment.")
    parser.add_argument("--goal", type=str, help="Natural language modelling objective.")
    parser.add_argument("--dataset", type=Path, help="Path to dataset file (csv/parquet/xlsx).")
    parser.add_argument("--target", type=str, default=None, help="Optional target column.")
    parser.add_argument(
        "--problem-type",
        choices=["classification", "regression"],
        default=None,
        help="Optional explicit problem type. Otherwise inferred from the goal and target.",
    )
    parser.add_argument("--program", type=Path, default=None, help="Optional JSON program spec.")
    parser.add_argument("--output", type=Path, default=Path("runs/latest"), help="Run directory.")
    parser.add_argument(
        "--description",
        type=str,
        default="baseline model sweep",
        help="Short note about what changed in this experiment.",
    )
    parser.add_argument("--test-size", type=float, default=None, help="Validation split size.")
    parser.add_argument("--cv-folds", type=int, default=None, help="Cross-validation folds on the training split.")
    parser.add_argument("--random-state", type=int, default=None, help="Random seed for the fixed split.")
    parser.add_argument(
        "--enable-deep-learning",
        dest="enable_deep_learning",
        action="store_true",
        help="Include PyTorch tabular candidates when torch is installed.",
    )
    parser.add_argument(
        "--disable-deep-learning",
        dest="enable_deep_learning",
        action="store_false",
        help="Disable deep learning candidates even if torch is installed.",
    )
    parser.set_defaults(enable_deep_learning=None)
    parser.add_argument(
        "--deep-learning-hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes for the PyTorch tabular MLP.",
    )
    parser.add_argument("--deep-learning-dropout", type=float, default=None, help="Dropout for the PyTorch tabular MLP.")
    parser.add_argument(
        "--deep-learning-learning-rate",
        type=float,
        default=None,
        help="Learning rate for the PyTorch tabular MLP.",
    )
    parser.add_argument(
        "--deep-learning-weight-decay",
        type=float,
        default=None,
        help="Weight decay for the PyTorch tabular MLP.",
    )
    parser.add_argument("--deep-learning-batch-size", type=int, default=None, help="Batch size for the PyTorch tabular MLP.")
    parser.add_argument("--deep-learning-max-epochs", type=int, default=None, help="Max epochs for the PyTorch tabular MLP.")
    parser.add_argument("--deep-learning-patience", type=int, default=None, help="Early stopping patience for the PyTorch tabular MLP.")
    parser.add_argument(
        "--deep-learning-validation-fraction",
        type=float,
        default=None,
        help="Internal validation fraction used by the PyTorch tabular MLP for early stopping.",
    )
    parser.add_argument(
        "--deep-learning-device",
        type=str,
        default=None,
        help="Device for PyTorch models: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--candidate-profile",
        choices=["balanced", "tree_heavy", "regularized", "deep_focus", "compact"],
        default=None,
        help="Preset that shifts the candidate hyperparameter family.",
    )
    return parser


def main() -> None:
    parser = buildParser()
    args = parser.parse_args()
    config = resolveConfig(args)
    prepared = prepareExperiment(config, args.output)
    summary = runExperiment(prepared, args.description)
    printSummary(summary)


if __name__ == "__main__":
    main()
