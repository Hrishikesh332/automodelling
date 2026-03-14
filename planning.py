from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Any

from deep_learning import isTorchAvailable
from prepare import ExperimentConfig

ALLOWED_PROFILES = ["balanced", "tree_heavy", "regularized", "deep_focus", "compact"]
ALLOWED_THRESHOLD_METRICS = {"balanced_accuracy", "f1", "recall"}


def buildSearchStrategy(preview: Any) -> dict[str, Any]:
    featurePlan = preview.feature_plan
    rowCount = int(preview.dataset_profile.get("rows", 0))
    numericCount = len(featurePlan.numeric_features)
    categoricalCount = len(featurePlan.categorical_features)
    highCardinalityCount = len(featurePlan.high_cardinality_columns)
    totalFeatures = max(1, numericCount + categoricalCount)
    numericShare = numericCount / totalFeatures
    mostlyNumeric = numericShare >= 0.75
    hasManyRows = rowCount >= 5000
    hasWideFeatureSpace = totalFeatures >= 40
    hasHeavyCategoricals = categoricalCount >= numericCount and categoricalCount >= 5
    hasHighCardinality = highCardinalityCount > 0
    recommendDeepLearning = (
        preview.config.enable_deep_learning
        and isTorchAvailable()
        and mostlyNumeric
        and (hasManyRows or hasWideFeatureSpace)
        and not hasHighCardinality
    )
    preferredProfile = "deep_focus" if recommendDeepLearning else "tree_heavy" if hasHeavyCategoricals or hasHighCardinality else "balanced"
    reasons: list[str] = []
    if recommendDeepLearning:
        reasons.append("Dataset is mostly numeric and large or wide enough for a PyTorch tabular model to be competitive.")
    if hasHeavyCategoricals:
        reasons.append("Categorical structure is strong, so tree ensembles should stay prominent.")
    if hasHighCardinality:
        reasons.append("High-cardinality features increase the value of tree-heavy search before promoting deep learning.")
    if not reasons:
        reasons.append("The dataset looks like a general mixed tabular problem, so a balanced search is appropriate.")

    return {
        "rowCount": rowCount,
        "numericCount": numericCount,
        "categoricalCount": categoricalCount,
        "highCardinalityCount": highCardinalityCount,
        "recommendDeepLearning": recommendDeepLearning,
        "preferredProfile": preferredProfile,
        "reasons": reasons,
    }


def buildHeuristicVariants(baseConfig: ExperimentConfig, preview: Any) -> list[dict[str, Any]]:
    strategy = buildSearchStrategy(preview)
    variants: list[dict[str, Any]] = []

    if strategy["preferredProfile"] == "deep_focus":
        variants.append(
            {
                "description": "deep-learning focused search",
                "reason": "Dataset understanding suggests a dense numeric problem where deep learning may win.",
                "changes": {"candidate_profile": "deep_focus"},
            }
        )
        variants.append(
            {
                "description": "baseline balanced search",
                "reason": "Keep a broad baseline so the deep-learning recommendation is tested against strong classical models.",
                "changes": {"candidate_profile": "balanced"},
            }
        )
        variants.append(
            {
                "description": "compact deep-learning search",
                "reason": "Check whether a smaller neural profile generalizes more cleanly on the same task.",
                "changes": {"candidate_profile": "compact"},
            }
        )
    else:
        variants.append(
            {
                "description": "baseline balanced search",
                "reason": "Establish a stable reference across all candidate families.",
                "changes": {"candidate_profile": "balanced"},
            }
        )
        variants.append(
            {
                "description": "tree-heavy search",
                "reason": "Dataset understanding suggests tree ensembles are likely to be strong.",
                "changes": {"candidate_profile": "tree_heavy"},
            }
        )
        variants.append(
            {
                "description": "regularized search",
                "reason": "Check whether a more conservative profile reduces overfitting.",
                "changes": {"candidate_profile": "regularized"},
            }
        )
        if baseConfig.enable_deep_learning and isTorchAvailable():
            variants.append(
                {
                    "description": "deep-learning challenger search",
                    "reason": "Run a neural challenger after the classical baselines to verify whether it adds value.",
                    "changes": {"candidate_profile": "deep_focus"},
                }
            )

    if preview.problem_type == "classification" and preview.isBinaryClassification:
        variants.append(
            {
                "description": "recall-aware threshold search",
                "reason": "Tune the decision threshold toward recall-sensitive binary behaviour.",
                "changes": {"binary_threshold_metric": "recall"},
            }
        )

    return variants


def plannerCommandValue(explicitCommand: str | None) -> str | None:
    if explicitCommand and explicitCommand.strip():
        return explicitCommand.strip()
    command = os.getenv("AUTOMODELLING_LLM_COMMAND", "").strip()
    return command or None


def llmPlannerReady(plannerMode: str, explicitCommand: str | None) -> bool:
    if plannerMode == "heuristic":
        return False
    return plannerCommandValue(explicitCommand) is not None


def variantSignature(changes: dict[str, Any]) -> str:
    return json.dumps(changes, sort_keys=True, ensure_ascii=True, default=str)


def usedVariantSignatures(executedRuns: list[dict[str, Any]]) -> set[str]:
    return {variantSignature(item.get("changes", {})) for item in executedRuns}


def buildPlannerContext(
    baseConfig: ExperimentConfig,
    preview: Any,
    executedRuns: list[dict[str, Any]],
    remainingBudget: int,
    heuristicVariants: list[dict[str, Any]],
) -> dict[str, Any]:
    strategy = buildSearchStrategy(preview)
    return {
        "task": "Suggest the next tabular modelling experiment to try.",
        "requirements": {
            "return_json_only": True,
            "must_include": ["description", "reason", "changes"],
            "allowed_change_fields": [
                "candidate_profile",
                "binary_threshold_metric",
                "enable_deep_learning",
                "deep_learning_hidden_dims",
                "deep_learning_dropout",
                "deep_learning_learning_rate",
                "deep_learning_weight_decay",
                "deep_learning_batch_size",
                "deep_learning_max_epochs",
                "deep_learning_patience",
            ],
            "allowed_candidate_profiles": ALLOWED_PROFILES,
            "allowed_binary_threshold_metrics": sorted(ALLOWED_THRESHOLD_METRICS),
        },
        "dataset": {
            "source": preview.dataset_source,
            "path": str(preview.dataset_path),
            "targetColumn": preview.target_column,
            "problemType": preview.problem_type,
            "primaryMetric": preview.primary_metric,
            "profile": preview.dataset_profile,
            "understanding": strategy,
        },
        "baseConfig": {
            "goal": baseConfig.goal,
            "candidateProfile": baseConfig.candidate_profile,
            "deepLearningEnabled": baseConfig.enable_deep_learning,
            "deepLearningHiddenDims": list(baseConfig.deep_learning_hidden_dims),
            "deepLearningDropout": baseConfig.deep_learning_dropout,
            "deepLearningLearningRate": baseConfig.deep_learning_learning_rate,
            "deepLearningWeightDecay": baseConfig.deep_learning_weight_decay,
            "deepLearningBatchSize": baseConfig.deep_learning_batch_size,
            "deepLearningMaxEpochs": baseConfig.deep_learning_max_epochs,
            "deepLearningPatience": baseConfig.deep_learning_patience,
        },
        "executedRuns": [
            {
                "experimentId": item["summary"]["experiment_id"],
                "description": item["summary"]["description"],
                "status": item["summary"]["status"],
                "bestModel": item["summary"]["best_candidate"]["name"],
                "primaryMetric": item["summary"]["primary_metric"],
                "primaryScore": item["summary"]["best_candidate"]["validation_metrics"].get(
                    item["summary"]["primary_metric"]
                ),
                "changes": item.get("changes", {}),
                "planner": item.get("planner", {}),
                "warnings": item["summary"].get("dataset_profile", {}).get("warnings", []),
            }
            for item in executedRuns
        ],
        "remainingBudget": remainingBudget,
        "heuristicSuggestions": heuristicVariants,
    }


def sanitizePlannerChanges(rawChanges: Any) -> dict[str, Any]:
    if not isinstance(rawChanges, dict):
        return {}

    changes: dict[str, Any] = {}
    candidateProfile = rawChanges.get("candidate_profile")
    if candidateProfile in ALLOWED_PROFILES:
        changes["candidate_profile"] = candidateProfile

    thresholdMetric = rawChanges.get("binary_threshold_metric")
    if thresholdMetric in ALLOWED_THRESHOLD_METRICS:
        changes["binary_threshold_metric"] = thresholdMetric

    enableDeepLearning = rawChanges.get("enable_deep_learning")
    if isinstance(enableDeepLearning, bool):
        changes["enable_deep_learning"] = enableDeepLearning

    hiddenDims = rawChanges.get("deep_learning_hidden_dims")
    if isinstance(hiddenDims, (list, tuple)):
        cleanedHiddenDims = []
        for value in hiddenDims[:5]:
            if isinstance(value, (int, float)) and int(value) > 0:
                cleanedHiddenDims.append(int(value))
        if cleanedHiddenDims:
            changes["deep_learning_hidden_dims"] = cleanedHiddenDims

    dropout = rawChanges.get("deep_learning_dropout")
    if isinstance(dropout, (int, float)) and 0.0 <= float(dropout) <= 0.8:
        changes["deep_learning_dropout"] = float(dropout)

    learningRate = rawChanges.get("deep_learning_learning_rate")
    if isinstance(learningRate, (int, float)) and 0.0 < float(learningRate) <= 1.0:
        changes["deep_learning_learning_rate"] = float(learningRate)

    weightDecay = rawChanges.get("deep_learning_weight_decay")
    if isinstance(weightDecay, (int, float)) and 0.0 <= float(weightDecay) <= 1.0:
        changes["deep_learning_weight_decay"] = float(weightDecay)

    batchSize = rawChanges.get("deep_learning_batch_size")
    if isinstance(batchSize, (int, float)) and 8 <= int(batchSize) <= 8192:
        changes["deep_learning_batch_size"] = int(batchSize)

    maxEpochs = rawChanges.get("deep_learning_max_epochs")
    if isinstance(maxEpochs, (int, float)) and 1 <= int(maxEpochs) <= 500:
        changes["deep_learning_max_epochs"] = int(maxEpochs)

    patience = rawChanges.get("deep_learning_patience")
    if isinstance(patience, (int, float)) and 1 <= int(patience) <= 100:
        changes["deep_learning_patience"] = int(patience)

    return changes


def parsePlannerOutput(output: str) -> dict[str, Any] | None:
    text = output.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                parsed = json.loads(line)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue
    return None


def runCommandPlanner(command: str, context: dict[str, Any]) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            shlex.split(command),
            input=json.dumps(context, indent=2, ensure_ascii=True),
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return parsePlannerOutput(completed.stdout)


def normalizePlannerVariant(
    response: dict[str, Any] | None,
    heuristicVariants: list[dict[str, Any]],
    command: str,
) -> dict[str, Any] | None:
    if response is None:
        return None

    heuristicIndex = response.get("heuristic_index")
    if isinstance(heuristicIndex, int) and 0 <= heuristicIndex < len(heuristicVariants):
        variant = dict(heuristicVariants[heuristicIndex])
        variant["planner"] = {
            "mode": "llm",
            "source": "command",
            "command": command,
            "selectionType": "heuristic_index",
            "rawResponse": response,
        }
        return variant

    changes = sanitizePlannerChanges(response.get("changes"))
    if not changes:
        return None

    description = str(response.get("description") or "llm-guided search").strip()
    reason = str(response.get("reason") or "Suggested by the external LLM planner.").strip()
    return {
        "description": description,
        "reason": reason,
        "changes": changes,
        "planner": {
            "mode": "llm",
            "source": "command",
            "command": command,
            "selectionType": "direct_changes",
            "rawResponse": response,
        },
    }


def nextHeuristicVariant(
    baseConfig: ExperimentConfig,
    preview: Any,
    executedRuns: list[dict[str, Any]],
) -> dict[str, Any] | None:
    usedSignatures = usedVariantSignatures(executedRuns)
    for variant in buildHeuristicVariants(baseConfig, preview):
        if variantSignature(variant["changes"]) in usedSignatures:
            continue
        selected = dict(variant)
        selected["planner"] = {
            "mode": "heuristic",
            "source": "built_in",
            "selectionType": "ordered_fallback",
        }
        return selected
    return None


def chooseNextVariant(
    baseConfig: ExperimentConfig,
    preview: Any,
    executedRuns: list[dict[str, Any]],
    maxExperiments: int,
    plannerMode: str,
    llmPlannerCommand: str | None,
) -> dict[str, Any] | None:
    remainingBudget = max(0, int(maxExperiments) - len(executedRuns))
    if remainingBudget <= 0:
        return None

    usedSignatures = usedVariantSignatures(executedRuns)
    heuristicVariants = buildHeuristicVariants(baseConfig, preview)
    command = plannerCommandValue(llmPlannerCommand)

    if llmPlannerReady(plannerMode, llmPlannerCommand) and command is not None:
        context = buildPlannerContext(baseConfig, preview, executedRuns, remainingBudget, heuristicVariants)
        response = runCommandPlanner(command, context)
        variant = normalizePlannerVariant(response, heuristicVariants, command)
        if variant is not None and variantSignature(variant["changes"]) not in usedSignatures:
            return variant

    return nextHeuristicVariant(baseConfig, preview, executedRuns)
