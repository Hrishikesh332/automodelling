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

    warnings = summary.get("dataset_profile", {}).get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
