from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from planning import buildSearchStrategy, chooseNextVariant, plannerCommandValue
from prepare import ExperimentConfig, ensureDir, prepareExperiment
from reporting import loadJsonIfPresent
from train import buildParser as buildTrainParser
from train import printSummary, resolveConfig, runExperiment


def runMode(argv: list[str]) -> None:
    parser = buildTrainParser()
    args = parser.parse_args(argv)
    config = resolveConfig(args)
    prepared = prepareExperiment(config, args.output)
    summary = runExperiment(prepared, args.description)
    printSummary(summary)


def applyConfigChanges(config: ExperimentConfig, changes: dict[str, Any]) -> ExperimentConfig:
    updated = deepcopy(config)
    for key, value in changes.items():
        setattr(updated, key, value)
    return updated


def writeSearchSummary(output_dir: Path, executed_runs: list[dict[str, Any]], preview: Any) -> Path:
    strategy = buildSearchStrategy(preview)
    path = output_dir / "agentic_search_summary.md"
    lines = [
        "# Agentic Search Summary",
        "",
        f"- Dataset source: {preview.dataset_source}",
        f"- Resolved dataset: {preview.dataset_path}",
        f"- Goal: {preview.config.goal}",
        f"- Target: {preview.target_column}",
        f"- Problem type: {preview.problem_type}",
        f"- Primary metric: {preview.primary_metric}",
        "",
        "## Dataset Understanding",
        f"- Rows: {strategy['rowCount']}",
        f"- Numeric features: {strategy['numericCount']}",
        f"- Categorical features: {strategy['categoricalCount']}",
        f"- High-cardinality categorical features: {strategy['highCardinalityCount']}",
        f"- Deep learning recommended: {strategy['recommendDeepLearning']}",
        f"- Preferred search profile: {strategy['preferredProfile']}",
    ]
    for reason in strategy["reasons"]:
        lines.append(f"- Reason: {reason}")
    lines.extend(
        [
            "",
        "## Experiments",
        ]
    )
    for item in executed_runs:
        summary = item["summary"]
        best = summary["best_candidate"]
        lines.extend(
            [
                f"### {summary['experiment_id']} - {summary['description']}",
                f"- Reason: {item['reason']}",
                f"- Planner: {item.get('planner', {}).get('mode', 'heuristic')}",
                f"- Status: {summary['status']}",
                f"- Best model: {best['name']}",
                f"- {summary['primary_metric']}: {best['validation_metrics'].get(summary['primary_metric'])}",
                f"- Ablation isolation: {summary.get('ablation', {}).get('isolation_level', '')}",
                f"- Experiment summary: {summary['artifacts']['experiment_summary']}",
                f"- Agent report: {summary['artifacts'].get('agent_report', '')}",
            ]
        )
        if summary["artifacts"].get("candidate_plot"):
            lines.append(f"- Candidate plot: {summary['artifacts']['candidate_plot']}")
        if summary["artifacts"].get("training_curve_plot"):
            lines.append(f"- Training curve: {summary['artifacts']['training_curve_plot']}")
        if summary["artifacts"].get("ablation_summary"):
            lines.append(f"- Ablation summary: {summary['artifacts']['ablation_summary']}")
        lines.append("")

    best_summary = loadJsonIfPresent(output_dir / "best_summary.json")
    if best_summary is not None:
        best = best_summary["best_candidate"]
        lines.extend(
            [
                "## Best Kept Artifact",
                f"- Experiment: {best_summary['experiment_id']}",
                f"- Model: {best['name']}",
                f"- {best_summary['primary_metric']}: {best['validation_metrics'].get(best_summary['primary_metric'])}",
                f"- Model artifact: {best_summary['artifacts'].get('best_model', '')}",
                f"- Best summary: {output_dir / 'best_summary.json'}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def writeAgentManifest(
    output_dir: Path,
    executed_runs: list[dict[str, Any]],
    preview: Any,
    plannerMode: str,
    llmPlannerCommand: str | None,
) -> Path:
    bestSummary = loadJsonIfPresent(output_dir / "best_summary.json")
    latestSummary = loadJsonIfPresent(output_dir / "latest_summary.json")
    strategy = buildSearchStrategy(preview)
    manifest = {
        "entrypoint": "python automodelling.py --dataset <path-or-url> --output <run-dir> [--target <column>]",
        "mode": "agentic_search",
        "datasetSource": preview.dataset_source,
        "resolvedDatasetPath": str(preview.dataset_path),
        "goal": preview.config.goal,
        "targetColumn": preview.target_column,
        "problemType": preview.problem_type,
        "primaryMetric": preview.primary_metric,
        "datasetUnderstanding": strategy,
        "bestKept": bestSummary,
        "latestRun": latestSummary,
        "experiments": [
            {
                "description": item["summary"]["description"],
                "reason": item["reason"],
                "status": item["summary"]["status"],
                "planner": item.get("planner", {}),
                "experimentId": item["summary"]["experiment_id"],
                "primaryScore": item["summary"]["best_candidate"]["validation_metrics"].get(
                    item["summary"]["primary_metric"]
                ),
                "summaryPath": item["summary"]["artifacts"]["experiment_summary"],
                "agentReportPath": item["summary"]["artifacts"].get("agent_report"),
                "ablationSummaryPath": item["summary"]["artifacts"].get("ablation_summary"),
            }
            for item in executed_runs
        ],
        "artifacts": {
            "resultsTsv": str(output_dir / "results.tsv"),
            "searchSummary": str(output_dir / "agentic_search_summary.md"),
            "bestSummary": str(output_dir / "best_summary.json"),
            "latestSummary": str(output_dir / "latest_summary.json"),
            "ablationTable": str(output_dir / "analysis" / "ablation_table.tsv"),
            "ablationSummary": str(output_dir / "analysis" / "ablation_summary.md"),
        },
        "planner": {
            "configuredMode": plannerMode,
            "llmCommandConfigured": plannerCommandValue(llmPlannerCommand) is not None,
        },
        "suggestedNextActions": [
            f"python automodelling.py inspect --output {output_dir}",
            f"python automodelling.py inspect --output {output_dir} --json",
            f"python automodelling.py agent --dataset {preview.dataset_source} --output {output_dir} --max-experiments 3",
        ],
    }
    path = output_dir / "agentic_manifest.json"
    ensureDir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True, default=str)
        handle.write("\n")
    return path


def searchMode(argv: list[str]) -> None:
    parser = buildTrainParser()
    parser.description = "Run an autonomous multi-experiment AutoModelling search."
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=5,
        help="Maximum autonomous search iterations to execute.",
    )
    parser.add_argument(
        "--search-planner",
        choices=["auto", "heuristic", "llm"],
        default="auto",
        help="How to choose the next experiment. 'llm' uses an external planner command and falls back safely if needed.",
    )
    parser.add_argument(
        "--llm-planner-command",
        type=str,
        default=None,
        help="Optional command that reads planner JSON from stdin and returns the next experiment as JSON.",
    )
    args = parser.parse_args(argv)
    base_config = resolveConfig(args)
    preview = prepareExperiment(deepcopy(base_config), args.output)
    base_config.goal = preview.config.goal

    executed_runs: list[dict[str, Any]] = []
    for _ in range(max(1, args.max_experiments)):
        variant = chooseNextVariant(
            base_config,
            preview,
            executed_runs,
            args.max_experiments,
            args.search_planner,
            args.llm_planner_command,
        )
        if variant is None:
            break
        config = applyConfigChanges(base_config, variant["changes"])
        prepared = prepareExperiment(config, args.output)
        summary = runExperiment(
            prepared,
            variant["description"],
            plannerContext=variant.get("planner"),
            plannedChanges=variant.get("changes"),
        )
        executed_runs.append(
            {
                "summary": summary,
                "reason": variant["reason"],
                "changes": variant["changes"],
                "planner": variant.get("planner", {}),
            }
        )
        printSummary(summary)

    search_summary_path = writeSearchSummary(args.output, executed_runs, preview)
    manifestPath = writeAgentManifest(
        args.output,
        executed_runs,
        preview,
        args.search_planner,
        args.llm_planner_command,
    )
    print(f"Agentic search summary: {search_summary_path}")
    print(f"Agentic manifest: {manifestPath}")


def buildInspectPayload(output_dir: Path, limit: int) -> dict[str, Any]:
    resultsPath = output_dir / "results.tsv"
    latestSummary = loadJsonIfPresent(output_dir / "latest_summary.json")
    bestSummary = loadJsonIfPresent(output_dir / "best_summary.json")
    searchSummaryPath = output_dir / "agentic_search_summary.md"
    manifestPath = output_dir / "agentic_manifest.json"

    if not resultsPath.exists() and latestSummary is None:
        raise SystemExit(f"No run artifacts found in: {output_dir}")

    rows: list[dict[str, str]] = []
    if resultsPath.exists():
        with resultsPath.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)

    return {
        "runDirectory": str(output_dir),
        "latestSummary": latestSummary,
        "bestSummary": bestSummary,
        "recentResults": rows[-max(1, limit) :] if rows else [],
        "searchSummaryPath": str(searchSummaryPath) if searchSummaryPath.exists() else None,
        "agenticManifestPath": str(manifestPath) if manifestPath.exists() else None,
        "ablationSummaryPath": str(output_dir / "analysis" / "ablation_summary.md")
        if (output_dir / "analysis" / "ablation_summary.md").exists()
        else None,
        "ablationTablePath": str(output_dir / "analysis" / "ablation_table.tsv")
        if (output_dir / "analysis" / "ablation_table.tsv").exists()
        else None,
    }


def inspectMode(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Inspect an existing AutoModelling run directory.")
    parser.add_argument("--output", type=Path, required=True, help="Run directory to inspect.")
    parser.add_argument("--limit", type=int, default=5, help="How many recent results to show.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary.")
    args = parser.parse_args(argv)

    payload = buildInspectPayload(args.output, args.limit)
    latestSummary = payload["latestSummary"]
    bestSummary = payload["bestSummary"]
    recentResults = payload["recentResults"]
    searchSummaryPath = payload["searchSummaryPath"]
    ablationSummaryPath = payload["ablationSummaryPath"]

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))
        return

    print(f"Run directory: {args.output}")
    if latestSummary is not None:
        latestBest = latestSummary["best_candidate"]
        print(
            "Latest: "
            f"{latestSummary['experiment_id']} ({latestSummary['status']}) "
            f"{latestSummary['primary_metric']}={latestBest['validation_metrics'].get(latestSummary['primary_metric'])}"
        )
        print(f"Latest best model: {latestBest['name']}")
        if latestBest.get("decision_threshold"):
            threshold = latestBest["decision_threshold"]
            print(
                "Latest threshold: "
                f"{threshold.get('threshold')} ({threshold.get('metric')}={threshold.get('score')})"
            )
        warnings = latestSummary.get("dataset_profile", {}).get("warnings", [])
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"- {warning}")
        artifacts = latestSummary.get("artifacts", {})
        planner = latestSummary.get("planner")
        if planner:
            print(f"Planner: {planner.get('mode')} ({planner.get('source')})")
        if artifacts.get("agent_report"):
            print(f"Agent report: {artifacts['agent_report']}")
        if artifacts.get("history_plot"):
            print(f"History plot: {artifacts['history_plot']}")
        if artifacts.get("candidate_plot"):
            print(f"Candidate plot: {artifacts['candidate_plot']}")
        if artifacts.get("training_curve_plot"):
            print(f"Training curve: {artifacts['training_curve_plot']}")
        if artifacts.get("model_card"):
            print(f"Model card: {artifacts['model_card']}")
        if artifacts.get("prediction_contract"):
            print(f"Prediction contract: {artifacts['prediction_contract']}")
        if artifacts.get("ablation_summary"):
            print(f"Ablation summary: {artifacts['ablation_summary']}")

    if bestSummary is not None:
        bestCandidate = bestSummary["best_candidate"]
        print(
            "Best kept: "
            f"{bestSummary['experiment_id']} "
            f"{bestSummary['primary_metric']}={bestCandidate['validation_metrics'].get(bestSummary['primary_metric'])}"
        )
        print(f"Best kept model: {bestCandidate['name']}")

    if searchSummaryPath is not None:
        print(f"Search summary: {searchSummaryPath}")
    if ablationSummaryPath is not None:
        print(f"Ablation summary: {ablationSummaryPath}")

    if recentResults:
        print("Recent results:")
        for row in recentResults:
            print(
                f"- {row['experiment_id']} {row['status']} "
                f"{row['primary_metric']}={row['primary_score']} "
                f"model={row['best_model']}"
            )


def initProgramMode(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Create a starter AutoModelling program JSON.")
    parser.add_argument("--path", type=Path, default=Path("program.json"), help="Where to write the JSON file.")
    parser.add_argument("--goal", type=str, default="", help="Optional natural language modelling goal.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path or URL.")
    parser.add_argument("--target", type=str, default=None, help="Optional explicit target column.")
    parser.add_argument("--problem-type", choices=["classification", "regression"], default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--drop-high-missing-threshold", type=float, default=0.98)
    parser.add_argument("--categorical-min-frequency", type=float, default=0.01)
    parser.add_argument("--categorical-max-categories", type=int, default=50)
    parser.add_argument("--numeric-clip-quantile", type=float, default=0.01)
    parser.add_argument("--binary-threshold-metric", type=str, default="balanced_accuracy")
    parser.add_argument("--enable-deep-learning", action="store_true", default=None)
    parser.add_argument("--disable-deep-learning", action="store_false", dest="enable_deep_learning")
    parser.add_argument("--deep-learning-hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--deep-learning-dropout", type=float, default=0.1)
    parser.add_argument("--deep-learning-learning-rate", type=float, default=1e-3)
    parser.add_argument("--deep-learning-weight-decay", type=float, default=1e-4)
    parser.add_argument("--deep-learning-batch-size", type=int, default=256)
    parser.add_argument("--deep-learning-max-epochs", type=int, default=30)
    parser.add_argument("--deep-learning-patience", type=int, default=6)
    parser.add_argument("--deep-learning-validation-fraction", type=float, default=0.15)
    parser.add_argument("--deep-learning-device", type=str, default="auto")
    parser.add_argument(
        "--candidate-profile",
        choices=["balanced", "tree_heavy", "regularized", "deep_focus", "compact"],
        default="balanced",
    )
    args = parser.parse_args(argv)

    payload = {
        "goal": args.goal,
        "dataset": args.dataset,
        "target": args.target,
        "problem_type": args.problem_type,
        "test_size": args.test_size,
        "cv_folds": args.cv_folds,
        "random_state": args.random_state,
        "drop_high_missing_threshold": args.drop_high_missing_threshold,
        "categorical_min_frequency": args.categorical_min_frequency,
        "categorical_max_categories": args.categorical_max_categories,
        "numeric_clip_quantile": args.numeric_clip_quantile,
        "binary_threshold_metric": args.binary_threshold_metric,
        "enable_deep_learning": args.enable_deep_learning,
        "deep_learning_hidden_dims": args.deep_learning_hidden_dims,
        "deep_learning_dropout": args.deep_learning_dropout,
        "deep_learning_learning_rate": args.deep_learning_learning_rate,
        "deep_learning_weight_decay": args.deep_learning_weight_decay,
        "deep_learning_batch_size": args.deep_learning_batch_size,
        "deep_learning_max_epochs": args.deep_learning_max_epochs,
        "deep_learning_patience": args.deep_learning_patience,
        "deep_learning_validation_fraction": args.deep_learning_validation_fraction,
        "deep_learning_device": args.deep_learning_device,
        "candidate_profile": args.candidate_profile,
    }
    payload = {key: value for key, value in payload.items() if value is not None}

    ensureDir(args.path.parent)
    with args.path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(f"Wrote starter program to: {args.path}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspectMode(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "init-program":
        initProgramMode(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        runMode(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        searchMode(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        searchMode(sys.argv[2:])
        return
    searchMode(sys.argv[1:])


if __name__ == "__main__":
    main()
