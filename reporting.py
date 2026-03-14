from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).joinpath("fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def getPyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def loadJsonIfPresent(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def jsonSafe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonSafe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonSafe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "get_params"):
        return {
            "estimator_class": value.__class__.__name__,
            "params": {key: jsonSafe(val) for key, val in value.get_params(deep=False).items()},
        }
    return str(value)


def estimatorSnapshot(estimator: Any) -> dict[str, Any]:
    return jsonSafe(estimator)


def writeJsonFile(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)
        handle.write("\n")


def summarizePlannedChangeKeys(plannedChanges: dict[str, Any]) -> list[str]:
    return sorted(str(key) for key in plannedChanges)


def describePlannedChanges(plannedChanges: dict[str, Any]) -> str:
    keys = summarizePlannedChangeKeys(plannedChanges)
    if not keys:
        return "baseline"
    return ", ".join(keys)


def plannerAssessment(summary: dict[str, Any], improvementEpsilon: float = 1e-6) -> dict[str, Any]:
    planner = summary.get("planner") or {"mode": "manual", "source": "manual"}
    previousBestModel = summary.get("comparison", {}).get("previous_best", {}).get("best_model")
    currentBestModel = summary.get("best_candidate", {}).get("name")
    delta = summary.get("comparison", {}).get("score_delta_vs_previous_best")
    if previousBestModel is None or delta is None:
        return {
            "planner_mode": planner.get("mode"),
            "planner_source": planner.get("source"),
            "winning_model_shift": "unmeasured",
            "note": "No previous kept best exists yet, so planner quality is not measurable on this run.",
        }

    if delta <= improvementEpsilon:
        return {
            "planner_mode": planner.get("mode"),
            "planner_source": planner.get("source"),
            "winning_model_shift": "not_supported",
            "note": "This run did not beat the previous kept best, so the planner choice is not supported by the tracked metric.",
        }

    if currentBestModel == previousBestModel:
        return {
            "planner_mode": planner.get("mode"),
            "planner_source": planner.get("source"),
            "winning_model_shift": "same_family",
            "note": f"Inference: the planner improved performance within the same winning model family ({currentBestModel}), so the gain looks more like better tuning or search than a family discovery.",
        }

    return {
        "planner_mode": planner.get("mode"),
        "planner_source": planner.get("source"),
        "winning_model_shift": "family_shift",
        "note": f"Inference: the planner found an improvement alongside a winning model-family change ({previousBestModel} -> {currentBestModel}).",
    }


def buildAblationMetadata(summary: dict[str, Any], improvementEpsilon: float = 1e-6) -> dict[str, Any]:
    plannedChanges = summary.get("planned_changes", {})
    plannedChangeKeys = summarizePlannedChangeKeys(plannedChanges)
    plannedChangeCount = len(plannedChangeKeys)
    comparison = summary.get("comparison", {})
    delta = comparison.get("score_delta_vs_previous_best")
    previousBestModel = comparison.get("previous_best", {}).get("best_model")
    currentBestModel = summary.get("best_candidate", {}).get("name")

    if plannedChangeCount == 0:
        isolation = "baseline"
        attribution = "Baseline run or manual rerun without an explicit planned change set."
    elif plannedChangeCount == 1:
        isolation = "single_change"
        field = plannedChangeKeys[0]
        if delta is None:
            attribution = f"Single planned change to `{field}`, but there is no previous kept best for comparison yet."
        elif delta > improvementEpsilon:
            attribution = f"Single planned change to `{field}` improved the tracked metric. This is the strongest causal evidence available in this harness."
        elif abs(delta) <= improvementEpsilon:
            attribution = f"Single planned change to `{field}` produced effectively no measurable improvement."
        else:
            attribution = f"Single planned change to `{field}` reduced the tracked metric."
    else:
        isolation = "bundled_changes"
        attribution = f"This run bundles {plannedChangeCount} planned changes, so attribution is confounded across: {describePlannedChanges(plannedChanges)}."

    if previousBestModel is None:
        familyNote = "No previous kept best exists yet, so model-family attribution is not available."
        bestModelChanged = None
    else:
        bestModelChanged = currentBestModel != previousBestModel
        if bestModelChanged:
            familyNote = f"The winning model changed from `{previousBestModel}` to `{currentBestModel}`."
        else:
            familyNote = f"The winning model stayed as `{currentBestModel}`, so any gain likely came from tuning or search rather than a family switch."

    plannerNote = plannerAssessment(summary, improvementEpsilon)
    return {
        "planned_change_count": plannedChangeCount,
        "planned_change_keys": plannedChangeKeys,
        "planned_changes_label": describePlannedChanges(plannedChanges),
        "isolation_level": isolation,
        "attribution_note": attribution,
        "best_model_changed_vs_previous_best": bestModelChanged,
        "family_note": familyNote,
        "planner_assessment": plannerNote,
    }


def flattenChanges(prefix: str, previous: Any, current: Any, changes: list[dict[str, Any]]) -> None:
    if isinstance(previous, dict) and isinstance(current, dict):
        keys = sorted(set(previous) | set(current))
        for key in keys:
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattenChanges(child_prefix, previous.get(key), current.get(key), changes)
        return
    if isinstance(previous, list) and isinstance(current, list):
        if previous != current:
            changes.append({"path": prefix, "previous": previous, "current": current})
        return
    if previous != current:
        changes.append({"path": prefix, "previous": previous, "current": current})


def diffDicts(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[dict[str, Any]]:
    if previous is None:
        return []
    changes: list[dict[str, Any]] = []
    flattenChanges("", previous, current, changes)
    return changes


def loadResultsRows(results_path: Path) -> list[dict[str, str]]:
    if not results_path.exists():
        return []
    with results_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def createHistoryPlot(output_dir: Path, primary_metric: str) -> Path | None:
    results_path = output_dir / "results.tsv"
    rows = loadResultsRows(results_path)
    if not rows:
        return None

    x_values = list(range(1, len(rows) + 1))
    scores = [float(row["primary_score"]) for row in rows if row.get("primary_score")]
    if len(scores) != len(rows):
        return None
    colors = ["#1b8a5a" if row.get("status") == "keep" else "#c16b00" for row in rows]

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "improvement_history.png"

    plt = getPyplot()
    plt.figure(figsize=(9, 4.8))
    plt.plot(x_values, scores, color="#1f3c88", linewidth=2, alpha=0.8)
    plt.scatter(x_values, scores, c=colors, s=80, zorder=3)
    plt.xlabel("Experiment")
    plt.ylabel(primary_metric)
    plt.title(f"Experiment History ({primary_metric})")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def createCandidatePlot(output_dir: Path, summary: dict[str, Any]) -> Path | None:
    candidates = summary.get("candidates", [])
    if not candidates:
        return None

    primary_metric = summary["primary_metric"]
    names = [candidate["name"] for candidate in candidates]
    scores = [
        candidate["validation_metrics"].get(primary_metric, float("nan"))
        for candidate in candidates
    ]
    if all(score != score for score in scores):
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{summary['experiment_id']}_candidate_scores.png"

    plt = getPyplot()
    plt.figure(figsize=(10, 5.2))
    bars = plt.bar(names, scores, color="#325c80")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(primary_metric)
    plt.title(f"Candidate Comparison for {summary['experiment_id']}")
    for bar, score in zip(bars, scores):
        if score == score:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def createTrainingCurvePlot(output_dir: Path, summary: dict[str, Any]) -> Path | None:
    history = summary.get("best_candidate", {}).get("training_history")
    if not history:
        return None
    epochs = history.get("epochs", [])
    if not epochs:
        return None

    x_values = [int(row["epoch"]) for row in epochs]
    train_loss = [float(row["train_loss"]) for row in epochs]
    val_loss = [float(row["val_loss"]) for row in epochs]

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{summary['experiment_id']}_training_curve.png"

    plt = getPyplot()
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(x_values, train_loss, label="train_loss", linewidth=2, color="#1b8a5a")
    plt.plot(x_values, val_loss, label="val_loss", linewidth=2, color="#c16b00")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve for {summary['best_candidate']['name']}")
    plt.legend()
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def generateVisualizations(output_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    try:
        history_path = createHistoryPlot(output_dir, summary["primary_metric"])
        candidate_path = createCandidatePlot(output_dir, summary)
        training_path = createTrainingCurvePlot(output_dir, summary)
    except Exception:
        return artifacts

    if history_path is not None:
        artifacts["history_plot"] = str(history_path)
    if candidate_path is not None:
        artifacts["candidate_plot"] = str(candidate_path)
    if training_path is not None:
        artifacts["training_curve_plot"] = str(training_path)
    return artifacts


def buildFeatureSchema(summary: dict[str, Any]) -> dict[str, Any]:
    datasetProfile = summary.get("dataset_profile", {})
    featurePlan = datasetProfile.get("feature_plan", {})
    dtypes = datasetProfile.get("dtypes", {})
    numericFeatures = set(featurePlan.get("numeric_features", []))
    categoricalFeatures = set(featurePlan.get("categorical_features", []))
    features = []
    for column in featurePlan.get("kept_columns", []):
        if column in numericFeatures:
            family = "numeric"
        elif column in categoricalFeatures:
            family = "categorical"
        else:
            family = "other"
        features.append(
            {
                "name": column,
                "dtype": dtypes.get(column),
                "family": family,
                "required": True,
            }
        )
    return {
        "goal": summary.get("goal"),
        "dataset": summary.get("dataset"),
        "targetColumn": summary.get("target_column"),
        "problemType": summary.get("problem_type"),
        "features": features,
    }


def predictionOutputColumns(summary: dict[str, Any]) -> list[str]:
    columns = ["prediction"]
    bestCandidate = summary.get("best_candidate", {})
    if summary.get("problem_type") == "classification":
        if bestCandidate.get("decision_threshold"):
            columns.append("decision_threshold")
            columns.append("threshold_metric")
        validationMetrics = bestCandidate.get("validation_metrics", {})
        if "roc_auc" in validationMetrics or "log_loss" in validationMetrics:
            columns.append("probability_columns_when_supported")
    return columns


def buildPredictionContract(summary: dict[str, Any]) -> dict[str, Any]:
    bestCandidate = summary.get("best_candidate", {})
    artifacts = summary.get("artifacts", {})
    return {
        "experimentId": summary.get("experiment_id"),
        "goal": summary.get("goal"),
        "problemType": summary.get("problem_type"),
        "targetColumn": summary.get("target_column"),
        "primaryMetric": summary.get("primary_metric"),
        "bestModel": bestCandidate.get("name"),
        "promotedModelArtifact": artifacts.get("best_model"),
        "validationPredictionsPath": bestCandidate.get("validation_predictions_path"),
        "requiredFeatureColumns": summary.get("dataset_profile", {}).get("feature_plan", {}).get("kept_columns", []),
        "predictionOutputs": predictionOutputColumns(summary),
        "decisionThreshold": bestCandidate.get("decision_threshold"),
    }


def buildModelCardLines(summary: dict[str, Any], promotedOnly: bool) -> list[str]:
    bestCandidate = summary.get("best_candidate", {})
    artifacts = summary.get("artifacts", {})
    qualityChecks = summary.get("dataset_profile", {}).get("quality_checks", {})
    featurePlan = summary.get("dataset_profile", {}).get("feature_plan", {})
    lines = [
        f"# {'Promoted' if promotedOnly else 'Latest'} Model Card",
        "",
        f"- Experiment: {summary.get('experiment_id')}",
        f"- Status: {summary.get('status')}",
        f"- Goal: {summary.get('goal')}",
        f"- Dataset: {summary.get('dataset')}",
        f"- Target: {summary.get('target_column')}",
        f"- Problem type: {summary.get('problem_type')}",
        f"- Model: {bestCandidate.get('name')}",
        f"- Primary metric: {summary.get('primary_metric')}={bestCandidate.get('validation_metrics', {}).get(summary.get('primary_metric'))}",
        f"- CV mean: {bestCandidate.get('cv_summary', {}).get('mean', {}).get(summary.get('primary_metric'))}",
        f"- Train feature count: {bestCandidate.get('train_feature_count')}",
        f"- Promoted model artifact: {artifacts.get('best_model', '')}",
        "",
        "## Input Features",
        f"- Total kept features: {len(featurePlan.get('kept_columns', []))}",
        f"- Numeric: {len(featurePlan.get('numeric_features', []))}",
        f"- Categorical: {len(featurePlan.get('categorical_features', []))}",
    ]
    if bestCandidate.get("decision_threshold"):
        threshold = bestCandidate["decision_threshold"]
        lines.append(
            f"- Decision threshold: {threshold.get('threshold')} ({threshold.get('metric')}={threshold.get('score')})"
        )
    warnings = summary.get("dataset_profile", {}).get("warnings", [])
    if warnings:
        lines.extend(["", "## Risks"])
        for warning in warnings:
            lines.append(f"- {warning}")
    if qualityChecks:
        lines.extend(
            [
                "",
                "## Quality Checks",
                f"- Duplicate feature rows: {qualityChecks.get('duplicate_feature_rows')}",
                f"- Conflicting duplicate target rows: {qualityChecks.get('conflicting_duplicate_target_rows')}",
                f"- Suspected leakage columns: {len(qualityChecks.get('suspected_leakage_columns', []))}",
            ]
        )
    lines.extend(
        [
            "",
            "## Key Artifacts",
            f"- Experiment summary: {artifacts.get('experiment_summary', '')}",
            f"- Validation predictions: {bestCandidate.get('validation_predictions_path', '')}",
        ]
    )
    return lines


def writeProductionArtifacts(output_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    handoffDir = output_dir / "handoff"
    handoffDir.mkdir(parents=True, exist_ok=True)
    featureSchemaPath = handoffDir / "feature_schema.json"
    predictionContractPath = handoffDir / "prediction_contract.json"
    latestModelCardPath = handoffDir / "latest_model_card.md"

    writeJsonFile(featureSchemaPath, buildFeatureSchema(summary))
    writeJsonFile(predictionContractPath, buildPredictionContract(summary))
    latestModelCardPath.write_text("\n".join(buildModelCardLines(summary, promotedOnly=False)) + "\n", encoding="utf-8")

    artifacts = {
        "feature_schema": str(featureSchemaPath),
        "prediction_contract": str(predictionContractPath),
        "model_card": str(latestModelCardPath),
    }

    if summary.get("artifacts", {}).get("best_model"):
        promotedContractPath = handoffDir / "promoted_prediction_contract.json"
        promotedModelCardPath = handoffDir / "promoted_model_card.md"
        writeJsonFile(promotedContractPath, buildPredictionContract(summary))
        promotedModelCardPath.write_text(
            "\n".join(buildModelCardLines(summary, promotedOnly=True)) + "\n",
            encoding="utf-8",
        )
        artifacts["promoted_prediction_contract"] = str(promotedContractPath)
        artifacts["promoted_model_card"] = str(promotedModelCardPath)

    return artifacts


def experimentIdSortKey(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(char for char in stem if char.isdigit())
    return (int(digits) if digits else 0, stem)


def loadExperimentSummaries(output_dir: Path) -> list[dict[str, Any]]:
    experimentsDir = output_dir / "experiments"
    if not experimentsDir.exists():
        return []
    summaries = []
    for path in sorted(experimentsDir.glob("exp_*.json"), key=experimentIdSortKey):
        with path.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    return summaries


def writeAblationArtifacts(output_dir: Path) -> dict[str, str]:
    summaries = loadExperimentSummaries(output_dir)
    if not summaries:
        return {}

    rows = []
    for summary in summaries:
        ablation = summary.get("ablation", {})
        planner = ablation.get("planner_assessment", {})
        comparison = summary.get("comparison", {})
        rows.append(
            {
                "experiment_id": summary.get("experiment_id"),
                "description": summary.get("description"),
                "status": summary.get("status"),
                "primary_metric": summary.get("primary_metric"),
                "primary_score": summary.get("best_candidate", {}).get("validation_metrics", {}).get(summary.get("primary_metric")),
                "delta_vs_previous_best": comparison.get("score_delta_vs_previous_best"),
                "planned_change_count": ablation.get("planned_change_count"),
                "planned_change_keys": ", ".join(ablation.get("planned_change_keys", [])),
                "isolation_level": ablation.get("isolation_level"),
                "best_model": summary.get("best_candidate", {}).get("name"),
                "previous_best_model": comparison.get("previous_best", {}).get("best_model"),
                "best_model_changed": ablation.get("best_model_changed_vs_previous_best"),
                "planner_mode": planner.get("planner_mode"),
                "planner_source": planner.get("planner_source"),
                "planner_shift_type": planner.get("winning_model_shift"),
                "attribution_note": ablation.get("attribution_note"),
                "planner_note": planner.get("note"),
            }
        )

    analysisDir = output_dir / "analysis"
    analysisDir.mkdir(parents=True, exist_ok=True)
    tablePath = analysisDir / "ablation_table.tsv"
    fieldnames = list(rows[0].keys())
    with tablePath.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    isolatedWins = [
        row for row in rows
        if row["isolation_level"] == "single_change"
        and isinstance(row["delta_vs_previous_best"], (int, float))
        and row["delta_vs_previous_best"] > 1e-6
    ]
    bundledWins = [
        row for row in rows
        if row["isolation_level"] == "bundled_changes"
        and isinstance(row["delta_vs_previous_best"], (int, float))
        and row["delta_vs_previous_best"] > 1e-6
    ]

    lines = [
        "# Ablation Summary",
        "",
        f"- Total experiments analysed: {len(rows)}",
        f"- Isolated winning ablations: {len(isolatedWins)}",
        f"- Bundled winning changes: {len(bundledWins)}",
        "",
        "## Interpretation",
        "- Single-change wins provide the strongest attribution evidence in this harness.",
        "- Bundled wins may still be useful, but they are confounded and should be followed by narrower ablations.",
        "",
        "## Isolated Evidence",
    ]
    if isolatedWins:
        isolatedWins = sorted(isolatedWins, key=lambda row: float(row["delta_vs_previous_best"]), reverse=True)
        for row in isolatedWins[:10]:
            lines.extend(
                [
                    f"### {row['experiment_id']} - {row['description']}",
                    f"- Delta vs previous best: {row['delta_vs_previous_best']}",
                    f"- Planned change: {row['planned_change_keys']}",
                    f"- Attribution: {row['attribution_note']}",
                    f"- Planner note: {row['planner_note']}",
                    "",
                ]
            )
    else:
        lines.append("- No isolated winning ablations yet.")
        lines.append("")

    lines.append("## Confounded Wins")
    if bundledWins:
        bundledWins = sorted(bundledWins, key=lambda row: float(row["delta_vs_previous_best"]), reverse=True)
        for row in bundledWins[:10]:
            lines.extend(
                [
                    f"### {row['experiment_id']} - {row['description']}",
                    f"- Delta vs previous best: {row['delta_vs_previous_best']}",
                    f"- Planned changes: {row['planned_change_keys']}",
                    f"- Attribution: {row['attribution_note']}",
                    "",
                ]
            )
    else:
        lines.append("- No bundled winning runs yet.")
        lines.append("")

    lines.append("## Next Actions")
    lines.append("- Prefer follow-up runs that change one planned field at a time when a bundled run wins.")
    lines.append(f"- Full table: {tablePath}")

    summaryPath = analysisDir / "ablation_summary.md"
    summaryPath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "ablation_table": str(tablePath),
        "ablation_summary": str(summaryPath),
    }


def writeAgentReport(path: Path, summary: dict[str, Any]) -> None:
    previous_best = summary.get("comparison", {}).get("previous_best", {})
    previous_latest = summary.get("comparison", {}).get("previous_latest", {})
    best_candidate = summary["best_candidate"]
    lines = [
        f"# Experiment {summary['experiment_id']}",
        "",
        f"- Status: {summary['status']}",
        f"- Description: {summary['description']}",
        f"- Best model: {best_candidate['name']}",
        f"- Primary metric: {summary['primary_metric']}={best_candidate['validation_metrics'].get(summary['primary_metric'])}",
    ]

    planner = summary.get("planner")
    if planner:
        lines.extend(
            [
                "",
                "## Planner",
                f"- Mode: {planner.get('mode')}",
                f"- Source: {planner.get('source')}",
                f"- Selection type: {planner.get('selectionType', '')}",
            ]
        )

    if summary.get("comparison", {}).get("score_delta_vs_previous_best") is not None:
        lines.append(
            f"- Delta vs previous best: {summary['comparison']['score_delta_vs_previous_best']}"
        )
    if summary.get("comparison", {}).get("score_delta_vs_previous_latest") is not None:
        lines.append(
            f"- Delta vs previous latest: {summary['comparison']['score_delta_vs_previous_latest']}"
        )

    lines.extend(["", "## Parameter Changes"])
    param_changes = summary.get("comparison", {}).get("parameter_changes_vs_previous_best", [])
    if param_changes:
        for change in param_changes[:30]:
            lines.append(
                f"- `{change['path']}`: `{change['previous']}` -> `{change['current']}`"
            )
    else:
        lines.append("- No parameter changes recorded relative to the previous kept best.")

    lines.extend(["", "## Config Changes"])
    config_changes = summary.get("comparison", {}).get("config_changes_vs_previous_best", [])
    if config_changes:
        for change in config_changes[:30]:
            lines.append(
                f"- `{change['path']}`: `{change['previous']}` -> `{change['current']}`"
            )
    else:
        lines.append("- No config changes recorded relative to the previous kept best.")

    lines.extend(["", "## Agent Recommendation"])
    if summary["status"] == "keep":
        lines.append("- Keep this run. It improved the tracked objective enough to replace the previous best.")
    else:
        lines.append("- Discard this run for promotion. Keep it only as evidence or an ablation point.")

    ablation = summary.get("ablation")
    if ablation:
        lines.extend(
            [
                "",
                "## Ablation",
                f"- Isolation level: {ablation.get('isolation_level')}",
                f"- Planned changes: {ablation.get('planned_changes_label')}",
                f"- Attribution: {ablation.get('attribution_note')}",
                f"- Family note: {ablation.get('family_note')}",
                f"- Planner note: {ablation.get('planner_assessment', {}).get('note')}",
            ]
        )

    warnings = summary.get("dataset_profile", {}).get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")

    artifacts = summary.get("artifacts", {})
    handoffKeys = [
        "model_card",
        "feature_schema",
        "prediction_contract",
        "promoted_model_card",
        "promoted_prediction_contract",
        "ablation_table",
        "ablation_summary",
    ]
    if any(artifacts.get(key) for key in handoffKeys):
        lines.extend(["", "## Production Handoff"])
        for key in handoffKeys:
            if artifacts.get(key):
                lines.append(f"- {key}: {artifacts[key]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
