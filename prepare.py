from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reporting import estimatorSnapshot

ProblemType = Literal["classification", "regression"]
LOWER_IS_BETTER = {"mae", "rmse", "median_ae", "log_loss", "brier_score"}
IMPROVEMENT_EPSILON = 1e-6
SUPPORTED_DATASET_EXTENSIONS = {".csv", ".parquet", ".pq", ".xlsx", ".xls"}
COMMON_TARGET_NAMES = [
    "target",
    "label",
    "class",
    "y",
    "outcome",
    "response",
    "churn",
    "default",
    "price",
    "sale_price",
    "revenue",
]
RESULTS_HEADER = [
    "created_at_utc",
    "experiment_id",
    "status",
    "description",
    "problem_type",
    "primary_metric",
    "primary_score",
    "cv_primary_mean",
    "cv_primary_std",
    "train_primary_score",
    "generalization_gap",
    "best_model",
    "fit_seconds",
    "cv_fit_seconds",
    "total_seconds",
    "train_rows",
    "validation_rows",
    "train_feature_count",
    "decision_threshold",
    "threshold_metric",
    "threshold_score",
    "accuracy",
    "balanced_accuracy",
    "f1_weighted",
    "precision_positive",
    "recall_positive",
    "roc_auc",
    "log_loss",
    "brier_score",
    "r2",
    "mae",
    "rmse",
    "median_ae",
]


@dataclass
class ExperimentConfig:
    goal: str = ""
    dataset: str = ""
    target: Optional[str] = None
    problem_type: Optional[ProblemType] = None
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    drop_high_missing_threshold: float = 0.98
    categorical_min_frequency: float = 0.01
    categorical_max_categories: int = 50
    numeric_clip_quantile: float = 0.01
    binary_threshold_metric: str = "balanced_accuracy"
    enable_deep_learning: bool = True
    deep_learning_hidden_dims: tuple[int, ...] = (256, 128)
    deep_learning_dropout: float = 0.1
    deep_learning_learning_rate: float = 1e-3
    deep_learning_weight_decay: float = 1e-4
    deep_learning_batch_size: int = 256
    deep_learning_max_epochs: int = 30
    deep_learning_patience: int = 6
    deep_learning_validation_fraction: float = 0.15
    deep_learning_device: str = "auto"
    candidate_profile: str = "balanced"


@dataclass
class FeaturePlan:
    kept_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_high_missing_columns: list[str] = field(default_factory=list)
    dropped_constant_columns: list[str] = field(default_factory=list)
    high_cardinality_columns: list[str] = field(default_factory=list)
    identifier_like_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreparedExperiment:
    config: ExperimentConfig
    output_dir: Path
    dataset_source: str
    dataset_path: Path
    target_column: str
    problem_type: ProblemType
    primary_metric: str
    cv_folds: int
    feature_plan: FeaturePlan
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    dataset_profile: dict[str, Any]

    @property
    def isBinaryClassification(self) -> bool:
        return self.problem_type == "classification" and self.y_train.nunique(dropna=True) == 2


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clips numeric columns to robust lower and upper quantiles."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: Any, y: Any = None) -> "QuantileClipper":
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        for idx in range(values.shape[1]):
            column = values[:, idx]
            finite_mask = np.isfinite(column)
            finite_values = column[finite_mask]
            if finite_values.size == 0:
                lower_bounds.append(float("-inf"))
                upper_bounds.append(float("inf"))
            else:
                lower_bounds.append(float(np.quantile(finite_values, self.lower_quantile)))
                upper_bounds.append(float(np.quantile(finite_values, self.upper_quantile)))
        self.lower_bounds_ = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds_ = np.asarray(upper_bounds, dtype=float)
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def ensureDir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safeJsonDump(data: dict[str, Any], path: Path) -> None:
    ensureDir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True, default=str)


def loadProgram(path: Path | None) -> ExperimentConfig | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ExperimentConfig(
        goal=raw.get("goal", ""),
        dataset=raw["dataset"],
        target=raw.get("target"),
        problem_type=raw.get("problem_type"),
        test_size=float(raw.get("test_size", 0.2)),
        cv_folds=int(raw.get("cv_folds", 5)),
        random_state=int(raw.get("random_state", 42)),
        drop_high_missing_threshold=float(raw.get("drop_high_missing_threshold", 0.98)),
        categorical_min_frequency=float(raw.get("categorical_min_frequency", 0.01)),
        categorical_max_categories=int(raw.get("categorical_max_categories", 50)),
        numeric_clip_quantile=float(raw.get("numeric_clip_quantile", 0.01)),
        binary_threshold_metric=str(raw.get("binary_threshold_metric", "balanced_accuracy")),
        enable_deep_learning=bool(raw.get("enable_deep_learning", True)),
        deep_learning_hidden_dims=tuple(int(value) for value in raw.get("deep_learning_hidden_dims", [256, 128])),
        deep_learning_dropout=float(raw.get("deep_learning_dropout", 0.1)),
        deep_learning_learning_rate=float(raw.get("deep_learning_learning_rate", 1e-3)),
        deep_learning_weight_decay=float(raw.get("deep_learning_weight_decay", 1e-4)),
        deep_learning_batch_size=int(raw.get("deep_learning_batch_size", 256)),
        deep_learning_max_epochs=int(raw.get("deep_learning_max_epochs", 30)),
        deep_learning_patience=int(raw.get("deep_learning_patience", 6)),
        deep_learning_validation_fraction=float(raw.get("deep_learning_validation_fraction", 0.15)),
        deep_learning_device=str(raw.get("deep_learning_device", "auto")),
        candidate_profile=str(raw.get("candidate_profile", "balanced")),
    )


def normalizeText(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def inferTargetFromGoal(goal: str, columns: list[str]) -> str | None:
    goal_n = normalizeText(goal)
    for column in columns:
        column_n = column.lower()
        patterns = [
            rf"\bpredict\s+{re.escape(column_n)}\b",
            rf"\bforecast\s+{re.escape(column_n)}\b",
            rf"\bestimate\s+{re.escape(column_n)}\b",
            rf"\btarget(?:\s+column)?\s*(?:is|=)\s*{re.escape(column_n)}\b",
            rf"\b{re.escape(column_n)}\s+is\s+the\s+target\b",
        ]
        if any(re.search(pattern, goal_n) for pattern in patterns):
            return column
    for column in columns:
        if re.search(rf"\b{re.escape(column.lower())}\b", goal_n):
            return column
    return None


def inferProblemType(goal: str, target_series: pd.Series) -> ProblemType:
    goal_n = normalizeText(goal)
    if any(
        keyword in goal_n
        for keyword in ["classify", "classification", "churn", "fraud", "default", "segment"]
    ):
        return "classification"
    if any(keyword in goal_n for keyword in ["regress", "regression", "forecast", "estimate", "sale price"]):
        return "regression"
    if pd.api.types.is_numeric_dtype(target_series):
        unique_values = target_series.nunique(dropna=True)
        return "classification" if unique_values <= 12 else "regression"
    return "classification"


def inferTargetWithoutGoal(columns: list[str]) -> tuple[str | None, str]:
    lowered = {column.lower(): column for column in columns}
    for candidate in COMMON_TARGET_NAMES:
        if candidate in lowered:
            return lowered[candidate], f"common_name:{candidate}"
    if columns:
        return columns[-1], "last_column"
    return None, "unavailable"


def selectTarget(goal: str, df: pd.DataFrame, explicit_target: str | None) -> tuple[str, str]:
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(f"Target column '{explicit_target}' not found in dataset.")
        return explicit_target, "explicit"
    inferred = inferTargetFromGoal(goal, df.columns.tolist()) if goal.strip() else None
    if inferred is None:
        inferred, method = inferTargetWithoutGoal(df.columns.tolist())
        if inferred is None:
            raise ValueError(
                "Could not infer target column. Pass --target, specify it in the program JSON, or provide a clearer goal."
            )
        return inferred, method
    return inferred, "goal"


def looksLikeUrl(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "file"}


def datasetFilenameFromSource(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme == "file":
        return Path(parsed.path).name
    if parsed.scheme in {"http", "https"}:
        return Path(parsed.path).name or "downloaded_dataset"
    return Path(source).name


def resolveDownloadPath(source: str, download_dir: Path) -> Path:
    name = datasetFilenameFromSource(source)
    suffix = Path(name).suffix
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    if suffix:
        return download_dir / f"{source_hash}{suffix}"
    return download_dir / f"{source_hash}.data"


def extractSupportedDataset(archive_path: Path, download_dir: Path) -> Path:
    extract_dir = download_dir / archive_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)

    candidates = sorted(
        path
        for path in extract_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DATASET_EXTENSIONS
    )
    if not candidates:
        raise ValueError(f"No supported dataset file found inside archive: {archive_path}")
    return candidates[0]


def resolveDatasetPath(dataset_source: str, output_dir: Path) -> Path:
    if looksLikeUrl(dataset_source):
        download_dir = output_dir / "_datasets"
        download_dir.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(dataset_source)
        if parsed.scheme == "file":
            local_path = Path(parsed.path)
            if not local_path.exists():
                raise ValueError(f"Dataset file does not exist: {local_path}")
            return local_path

        destination = resolveDownloadPath(dataset_source, download_dir)
        if not destination.exists():
            with urlopen(dataset_source) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if destination.suffix.lower() == ".zip":
            return extractSupportedDataset(destination, download_dir)
        return destination
    return Path(dataset_source)


def loadDataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for loader in (pd.read_csv, pd.read_parquet, pd.read_excel):
        try:
            return loader(path)
        except Exception:
            continue
    raise ValueError(f"Unsupported dataset extension: {suffix}")


def buildFeaturePlan(X: pd.DataFrame, config: ExperimentConfig) -> FeaturePlan:
    missing_ratio = X.isna().mean()
    high_missing_columns = sorted(
        [column for column, ratio in missing_ratio.items() if ratio >= config.drop_high_missing_threshold]
    )

    constant_columns: list[str] = []
    kept_candidates = [column for column in X.columns if column not in high_missing_columns]
    for column in kept_candidates:
        if X[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)

    kept_columns = [column for column in X.columns if column not in high_missing_columns and column not in constant_columns]
    planned = X[kept_columns]
    numeric_features = planned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in kept_columns if column not in numeric_features]

    high_cardinality_columns = sorted(
        [
            column
            for column in categorical_features
            if planned[column].nunique(dropna=True) > max(config.categorical_max_categories, 100)
        ]
    )

    identifier_like_columns: list[str] = []
    for column in kept_columns:
        unique_count = planned[column].nunique(dropna=True)
        unique_ratio = unique_count / max(1, len(planned))
        name_looks_like_id = bool(re.search(r"(^id$|_id$|id_|uuid|guid|key|identifier)", column.lower()))
        dtype_is_non_numeric = not pd.api.types.is_numeric_dtype(planned[column])
        if unique_ratio >= 0.98 and unique_count >= max(50, int(0.5 * len(planned))) and (
            name_looks_like_id or dtype_is_non_numeric
        ):
            identifier_like_columns.append(column)
    identifier_like_columns = sorted(identifier_like_columns)

    warnings: list[str] = []
    if high_missing_columns:
        warnings.append(
            f"Dropping {len(high_missing_columns)} feature(s) with >= {config.drop_high_missing_threshold:.0%} missingness."
        )
    if constant_columns:
        warnings.append(f"Dropping {len(constant_columns)} constant feature(s).")
    if high_cardinality_columns:
        warnings.append(
            "High-cardinality categorical features detected; infrequent categories will be grouped during encoding."
        )
    if identifier_like_columns:
        warnings.append(
            "Identifier-like columns detected. Review them for leakage before trusting leaderboard gains."
        )

    return FeaturePlan(
        kept_columns=kept_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_high_missing_columns=high_missing_columns,
        dropped_constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        identifier_like_columns=identifier_like_columns,
        warnings=warnings,
    )


def buildOneHotEncoder(config: ExperimentConfig) -> OneHotEncoder:
    encoder_kwargs = {
        "handle_unknown": "infrequent_if_exist",
        "min_frequency": config.categorical_min_frequency,
        "max_categories": config.categorical_max_categories,
    }
    try:
        return OneHotEncoder(sparse_output=False, **encoder_kwargs)
    except TypeError:
        try:
            return OneHotEncoder(sparse=False, **encoder_kwargs)
        except TypeError:
            fallback = dict(encoder_kwargs)
            fallback.pop("min_frequency", None)
            fallback.pop("max_categories", None)
            fallback["handle_unknown"] = "ignore"
            return OneHotEncoder(sparse=False, **fallback)


def buildPreprocessor(feature_plan: FeaturePlan, config: ExperimentConfig) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("clipper", QuantileClipper(config.numeric_clip_quantile, 1.0 - config.numeric_clip_quantile)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", buildOneHotEncoder(config)),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if feature_plan.numeric_features:
        transformers.append(("num", numeric_pipeline, feature_plan.numeric_features))
    if feature_plan.categorical_features:
        transformers.append(("cat", categorical_pipeline, feature_plan.categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def defaultPrimaryMetric(problem_type: ProblemType, target: pd.Series) -> str:
    if problem_type == "classification":
        return "roc_auc" if target.nunique(dropna=True) == 2 else "f1_weighted"
    return "r2"


def buildDatasetProfile(
    df: pd.DataFrame,
    target_column: str,
    problem_type: ProblemType,
    feature_plan: FeaturePlan,
    quality_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing_pct = (df.isna().mean() * 100.0).round(4).to_dict()
    dtypes = {column: str(dtype) for column, dtype in df.dtypes.to_dict().items()}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        numeric_summary_df = df[numeric_columns].describe().transpose().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        numeric_summary = {
            column: {key: float(value) for key, value in row.to_dict().items()}
            for column, row in numeric_summary_df.iterrows()
        }
    else:
        numeric_summary = {}

    target_series = df[target_column]
    target_summary: dict[str, Any]
    if problem_type == "classification":
        value_counts = target_series.fillna("<NA>").astype(str).value_counts(dropna=False)
        counts_dict = {str(label): int(count) for label, count in value_counts.head(20).items()}
        if value_counts.empty:
            imbalance_ratio = 1.0
        else:
            imbalance_ratio = float(value_counts.max() / max(1, value_counts.min()))
        target_summary = {
            "distribution": counts_dict,
            "class_imbalance_ratio": imbalance_ratio,
        }
    else:
        describe = target_series.describe().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target_summary = {"distribution": {key: float(value) for key, value in describe.to_dict().items()}}

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_column,
        "problem_type": problem_type,
        "missing_pct": missing_pct,
        "dtypes": dtypes,
        "numeric_summary": numeric_summary,
        "cardinality": {column: int(df[column].nunique(dropna=True)) for column in df.columns},
        "target_summary": target_summary,
        "feature_plan": asdict(feature_plan),
        "warnings": feature_plan.warnings,
        "quality_checks": quality_checks or {},
    }


def stringifiedSeries(series: pd.Series) -> pd.Series:
    return series.fillna("<NA>").astype(str).str.strip().str.lower()


def detectLeakageColumns(
    X: pd.DataFrame,
    y: pd.Series,
) -> list[dict[str, Any]]:
    suspected: list[dict[str, Any]] = []
    targetText = stringifiedSeries(y)

    for column in X.columns:
        columnSeries = X[column]
        featureText = stringifiedSeries(columnSeries)
        equalityShare = float((featureText == targetText).mean())
        numericCorrelation: float | None = None
        if pd.api.types.is_numeric_dtype(columnSeries) and pd.api.types.is_numeric_dtype(y):
            joined = pd.concat(
                [
                    pd.to_numeric(columnSeries, errors="coerce").rename("feature"),
                    pd.to_numeric(y, errors="coerce").rename("target"),
                ],
                axis=1,
            ).dropna()
            if len(joined) >= 3 and joined["feature"].nunique() > 1 and joined["target"].nunique() > 1:
                numericCorrelation = float(abs(joined["feature"].corr(joined["target"])))

        if equalityShare >= 0.98 or (numericCorrelation is not None and numericCorrelation >= 0.995):
            suspected.append(
                {
                    "column": column,
                    "equality_share": round(equalityShare, 6),
                    "numeric_correlation": None if numericCorrelation is None else round(numericCorrelation, 6),
                }
            )

    return suspected


def detectConflictingDuplicateTargets(X: pd.DataFrame, y: pd.Series) -> int:
    if X.empty:
        return 0
    rowHashes = pd.util.hash_pandas_object(X.fillna("<NA>"), index=False)
    grouped = pd.DataFrame({"row_hash": rowHashes, "target": y}).groupby("row_hash")["target"].nunique(dropna=False)
    conflictingHashes = grouped[grouped > 1].index
    if len(conflictingHashes) == 0:
        return 0
    return int(pd.Series(rowHashes).isin(conflictingHashes).sum())


def buildSplitDrift(
    problem_type: ProblemType,
    y_train: pd.Series,
    y_val: pd.Series,
) -> dict[str, Any]:
    if problem_type == "classification":
        trainDist = (y_train.fillna("<NA>").astype(str).value_counts(normalize=True)).to_dict()
        valDist = (y_val.fillna("<NA>").astype(str).value_counts(normalize=True)).to_dict()
        labels = sorted(set(trainDist) | set(valDist))
        maxAbsShareDelta = 0.0
        for label in labels:
            maxAbsShareDelta = max(maxAbsShareDelta, abs(trainDist.get(label, 0.0) - valDist.get(label, 0.0)))
        return {
            "train_distribution": {str(key): float(value) for key, value in trainDist.items()},
            "validation_distribution": {str(key): float(value) for key, value in valDist.items()},
            "max_abs_share_delta": float(maxAbsShareDelta),
        }

    trainMean = float(pd.to_numeric(y_train, errors="coerce").mean())
    valMean = float(pd.to_numeric(y_val, errors="coerce").mean())
    trainStd = float(pd.to_numeric(y_train, errors="coerce").std(ddof=0) or 0.0)
    valStd = float(pd.to_numeric(y_val, errors="coerce").std(ddof=0) or 0.0)
    denominator = max(abs(trainStd), 1e-12)
    return {
        "train_mean": trainMean,
        "validation_mean": valMean,
        "train_std": trainStd,
        "validation_std": valStd,
        "mean_shift_std_units": float(abs(trainMean - valMean) / denominator),
    }


def buildQualityChecks(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: ProblemType,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[dict[str, Any], list[str]]:
    duplicateFeatureRows = int(X.duplicated().sum())
    conflictingDuplicateTargets = detectConflictingDuplicateTargets(X, y)
    suspectedLeakageColumns = detectLeakageColumns(X, y)
    splitDrift = buildSplitDrift(problem_type, y_train, y_val)

    warnings: list[str] = []
    if duplicateFeatureRows > 0:
        warnings.append(
            f"Detected {duplicateFeatureRows} duplicate feature row(s). Review whether repeated entities are expected."
        )
    if conflictingDuplicateTargets > 0:
        warnings.append(
            f"Detected {conflictingDuplicateTargets} row(s) where identical features map to different targets."
        )
    if suspectedLeakageColumns:
        columns = ", ".join(item["column"] for item in suspectedLeakageColumns[:5])
        warnings.append(
            f"Possible target leakage detected in: {columns}. Review these columns before trusting leaderboard gains."
        )

    if problem_type == "classification":
        if float(splitDrift.get("max_abs_share_delta", 0.0)) > 0.15:
            warnings.append("Train/validation class balance drift looks high. Review the split before trusting the score.")
    elif float(splitDrift.get("mean_shift_std_units", 0.0)) > 0.5:
        warnings.append("Train/validation target distribution drift looks high for regression. Review the split.")

    return (
        {
            "duplicate_feature_rows": duplicateFeatureRows,
            "conflicting_duplicate_target_rows": conflictingDuplicateTargets,
            "suspected_leakage_columns": suspectedLeakageColumns,
            "split_drift": splitDrift,
        },
        warnings,
    )


def resolveCvFolds(problem_type: ProblemType, y_train: pd.Series, requested_folds: int) -> int:
    if requested_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if problem_type == "classification":
        min_class_count = int(y_train.value_counts(dropna=False).min())
        resolved = min(requested_folds, min_class_count)
    else:
        resolved = min(requested_folds, len(y_train))
    if resolved < 2:
        raise ValueError("Not enough rows per split to run cross-validation. Lower cv_folds or use more data.")
    return resolved


def prepareExperiment(config: ExperimentConfig, output_dir: Path) -> PreparedExperiment:
    ensureDir(output_dir)
    dataset_source = config.dataset
    dataset_path = resolveDatasetPath(dataset_source, output_dir)
    df = loadDataset(dataset_path)

    target_column, target_inference_method = selectTarget(config.goal, df, config.target)
    original_row_count = int(df.shape[0])
    df = df.loc[df[target_column].notna()].copy()
    dropped_target_na = original_row_count - int(df.shape[0])
    if df.empty:
        raise ValueError("Dataset is empty after dropping rows with missing target values.")

    target_series = df[target_column]
    if config.problem_type is not None:
        if config.problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem_type '{config.problem_type}'.")
        problem_type: ProblemType = config.problem_type
        problem_type_inference_method = "explicit"
    else:
        problem_type = inferProblemType(config.goal, target_series)
        problem_type_inference_method = "goal_or_dtype"
    primary_metric = defaultPrimaryMetric(problem_type, target_series)
    if not config.goal.strip():
        verb = "Predict" if problem_type == "classification" else "Estimate"
        config.goal = f"{verb} {target_column}"
        goal_inference_method = "auto_generated"
    else:
        goal_inference_method = "provided"

    X_full = df.drop(columns=[target_column])
    feature_plan = buildFeaturePlan(X_full, config)
    X = X_full[feature_plan.kept_columns].copy()
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after preprocessing rules.")
    y = df[target_column]

    stratify = y if problem_type == "classification" and y.nunique(dropna=True) > 1 else None
    used_stratify = stratify is not None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )
    except ValueError:
        used_stratify = False
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=None,
        )

    cv_folds = resolveCvFolds(problem_type, y_train, config.cv_folds)
    qualityChecks, qualityWarnings = buildQualityChecks(X, y, problem_type, y_train, y_val)
    feature_plan.warnings.extend(qualityWarnings)
    dataset_profile = buildDatasetProfile(df, target_column, problem_type, feature_plan, quality_checks=qualityChecks)
    dataset_profile["rows_dropped_missing_target"] = dropped_target_na
    dataset_profile["dataset_source"] = dataset_source
    dataset_profile["resolved_dataset_path"] = str(dataset_path)
    dataset_profile["inference"] = {
        "target_column_method": target_inference_method,
        "problem_type_method": problem_type_inference_method,
        "goal_method": goal_inference_method,
    }
    dataset_profile["split"] = {
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_size": float(config.test_size),
        "random_state": int(config.random_state),
        "stratified": used_stratify,
        "cv_folds": int(cv_folds),
        "primary_metric": primary_metric,
        "binary_threshold_metric": config.binary_threshold_metric if problem_type == "classification" else None,
        "enable_deep_learning": config.enable_deep_learning,
    }
    safeJsonDump(dataset_profile, output_dir / "dataset_profile.json")

    return PreparedExperiment(
        config=config,
        output_dir=output_dir,
        dataset_source=dataset_source,
        dataset_path=dataset_path,
        target_column=target_column,
        problem_type=problem_type,
        primary_metric=primary_metric,
        cv_folds=cv_folds,
        feature_plan=feature_plan,
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        dataset_profile=dataset_profile,
    )


def makeCvSplitter(problem_type: ProblemType, cv_folds: int, random_state: int) -> Any:
    if problem_type == "classification":
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def probabilityScores(probabilities: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if probabilities is None or probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None
    return probabilities[:, 1]


def thresholdPredictions(classes: np.ndarray, positive_scores: np.ndarray, threshold: float) -> np.ndarray:
    negative_class, positive_class = classes[0], classes[1]
    return np.where(positive_scores >= threshold, positive_class, negative_class)


def predictModelOutputs(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    decision_scores = model.decision_function(X) if hasattr(model, "decision_function") else None
    return predictions, probabilities, decision_scores


def evaluatePredictions(
    problem_type: ProblemType,
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    decision_scores: Optional[np.ndarray] = None,
    positive_label: Any = None,
) -> dict[str, float]:
    if problem_type == "classification":
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "f1_weighted": float(f1_score(y_true, predictions, average="weighted")),
        }

        unique_classes = int(pd.Series(y_true).nunique(dropna=True))
        auc_input: Optional[np.ndarray] = None
        if probabilities is not None:
            try:
                metrics["log_loss"] = float(log_loss(y_true, probabilities))
            except ValueError:
                pass
            if unique_classes == 2 and probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                auc_input = probabilities[:, 1]
                if positive_label is not None:
                    y_binary = (pd.Series(y_true).to_numpy() == positive_label).astype(int)
                    metrics["brier_score"] = float(brier_score_loss(y_binary, probabilities[:, 1]))
            elif unique_classes > 2 and probabilities.ndim == 2:
                auc_input = probabilities
        elif decision_scores is not None:
            auc_input = decision_scores

        if unique_classes == 2 and positive_label is not None:
            metrics["precision_positive"] = float(
                precision_score(y_true, predictions, pos_label=positive_label, zero_division=0)
            )
            metrics["recall_positive"] = float(
                recall_score(y_true, predictions, pos_label=positive_label, zero_division=0)
            )

        if auc_input is not None:
            try:
                if unique_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, auc_input))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, auc_input, multi_class="ovr", average="weighted")
                    )
            except ValueError:
                pass
        return metrics

    return {
        "r2": float(r2_score(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, predictions))),
        "median_ae": float(median_absolute_error(y_true, predictions)),
    }


def summarizeMetricCollection(metric_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {"mean": {}, "std": {}}
    metric_names = sorted({key for row in metric_rows for key in row})
    for metric_name in metric_names:
        values = [row[metric_name] for row in metric_rows if metric_name in row]
        summary["mean"][metric_name] = float(np.mean(values))
        summary["std"][metric_name] = float(np.std(values))
    return summary


def metricGap(metric_name: str, train_value: Optional[float], validation_value: Optional[float]) -> Optional[float]:
    if train_value is None or validation_value is None:
        return None
    if metric_name in LOWER_IS_BETTER:
        return float(validation_value - train_value)
    return float(train_value - validation_value)


def thresholdObjectiveScore(metric_name: str, y_true: pd.Series, predictions: np.ndarray, positive_label: Any) -> float:
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, predictions))
    if metric_name == "f1":
        return float(f1_score(y_true, predictions, pos_label=positive_label, zero_division=0))
    if metric_name == "recall":
        return float(recall_score(y_true, predictions, pos_label=positive_label, zero_division=0))
    return float(balanced_accuracy_score(y_true, predictions))


def optimizeBinaryThreshold(
    y_true: pd.Series,
    positive_scores: np.ndarray,
    positive_label: Any,
    metric_name: str,
) -> dict[str, Any]:
    threshold_candidates = sorted(set(np.linspace(0.05, 0.95, 37).tolist() + [0.5]))
    classes = np.asarray([label for label in pd.Series(y_true).dropna().unique() if label != positive_label] + [positive_label], dtype=object)
    if len(classes) != 2:
        return {}

    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in threshold_candidates:
        predictions = thresholdPredictions(classes, positive_scores, threshold)
        score = thresholdObjectiveScore(metric_name, y_true, predictions, positive_label)
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = threshold
    return {
        "threshold": float(best_threshold),
        "metric": metric_name,
        "score": float(best_score),
        "positive_label": positive_label,
    }


def transformedFeatureCount(model: Pipeline, X: pd.DataFrame) -> int:
    sample = X.iloc[:1]
    transformed = model.named_steps["preprocess"].transform(sample)
    if hasattr(transformed, "shape"):
        return int(transformed.shape[1])
    return int(np.asarray(transformed).shape[1])


def evaluateCandidate(
    prepared: PreparedExperiment,
    estimator: Any,
) -> dict[str, Any]:
    cv_splitter = makeCvSplitter(prepared.problem_type, prepared.cv_folds, prepared.config.random_state)
    cv_rows: list[dict[str, float]] = []
    cv_fit_seconds = 0.0
    threshold_details: dict[str, Any] | None = None
    oof_positive_scores = np.full(len(prepared.X_train), np.nan, dtype=float)
    positive_label: Any = None

    splitter_input = prepared.y_train if prepared.problem_type == "classification" else None
    for train_idx, fold_idx in cv_splitter.split(prepared.X_train, splitter_input):
        X_fold_train = prepared.X_train.iloc[train_idx]
        y_fold_train = prepared.y_train.iloc[train_idx]
        X_fold_val = prepared.X_train.iloc[fold_idx]
        y_fold_val = prepared.y_train.iloc[fold_idx]

        fold_model = Pipeline(
            steps=[
                ("preprocess", buildPreprocessor(prepared.feature_plan, prepared.config)),
                ("model", clone(estimator)),
            ]
        )
        start = perf_counter()
        fold_model.fit(X_fold_train, y_fold_train)
        cv_fit_seconds += perf_counter() - start

        predictions, probabilities, decision_scores = predictModelOutputs(fold_model, X_fold_val)
        fold_positive_label = None
        if prepared.isBinaryClassification and probabilities is not None:
            fold_classes = getattr(fold_model.named_steps["model"], "classes_", None)
            if fold_classes is not None and len(fold_classes) == 2:
                positive_label = fold_classes[1]
                fold_positive_label = positive_label
                positive_scores = probabilityScores(probabilities)
                if positive_scores is not None:
                    oof_positive_scores[fold_idx] = positive_scores
        cv_rows.append(
            evaluatePredictions(
                prepared.problem_type,
                y_fold_val,
                predictions,
                probabilities=probabilities,
                decision_scores=decision_scores,
                positive_label=fold_positive_label,
            )
        )

    if prepared.isBinaryClassification and positive_label is not None and not np.isnan(oof_positive_scores).any():
        threshold_details = optimizeBinaryThreshold(
            prepared.y_train,
            oof_positive_scores,
            positive_label,
            prepared.config.binary_threshold_metric,
        )

    cv_summary = summarizeMetricCollection(cv_rows)

    final_model = Pipeline(
        steps=[
            ("preprocess", buildPreprocessor(prepared.feature_plan, prepared.config)),
            ("model", clone(estimator)),
        ]
    )
    fit_start = perf_counter()
    final_model.fit(prepared.X_train, prepared.y_train)
    fit_seconds = perf_counter() - fit_start
    trained_estimator = final_model.named_steps["model"]

    final_classes = getattr(final_model.named_steps["model"], "classes_", None)
    if prepared.isBinaryClassification and final_classes is not None and len(final_classes) == 2:
        positive_label = final_classes[1]
        if threshold_details is not None:
            threshold_details["positive_label"] = positive_label

    train_predictions, train_probabilities, train_decision_scores = predictModelOutputs(final_model, prepared.X_train)
    if threshold_details is not None and train_probabilities is not None and positive_label is not None:
        train_scores = probabilityScores(train_probabilities)
        if train_scores is not None:
            train_predictions = thresholdPredictions(final_classes, train_scores, threshold_details["threshold"])
    train_metrics = evaluatePredictions(
        prepared.problem_type,
        prepared.y_train,
        train_predictions,
        probabilities=train_probabilities,
        decision_scores=train_decision_scores,
        positive_label=positive_label,
    )

    validation_predictions, validation_probabilities, validation_decision_scores = predictModelOutputs(
        final_model,
        prepared.X_val,
    )
    if threshold_details is not None and validation_probabilities is not None and positive_label is not None:
        validation_scores = probabilityScores(validation_probabilities)
        if validation_scores is not None:
            validation_predictions = thresholdPredictions(
                final_classes,
                validation_scores,
                threshold_details["threshold"],
            )
    validation_metrics = evaluatePredictions(
        prepared.problem_type,
        prepared.y_val,
        validation_predictions,
        probabilities=validation_probabilities,
        decision_scores=validation_decision_scores,
        positive_label=positive_label,
    )

    return {
        "model": final_model,
        "estimator_class": trained_estimator.__class__.__name__,
        "estimator_params": estimatorSnapshot(trained_estimator),
        "cv_rows": cv_rows,
        "cv_summary": cv_summary,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "fit_seconds": float(fit_seconds),
        "cv_fit_seconds": float(cv_fit_seconds),
        "generalization_gap": metricGap(
            prepared.primary_metric,
            train_metrics.get(prepared.primary_metric),
            validation_metrics.get(prepared.primary_metric),
        ),
        "decision_threshold": threshold_details,
        "train_feature_count": transformedFeatureCount(final_model, prepared.X_train),
        "training_history": getattr(trained_estimator, "training_history_", None),
    }


def primaryScore(result: dict[str, Any], primary_metric: str) -> Optional[float]:
    return result["validation_metrics"].get(primary_metric)


def nextExperimentId(results_path: Path) -> str:
    if not results_path.exists():
        return "exp_0001"
    with results_path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for line in handle if line.strip())
    data_rows = max(0, row_count - 1)
    return f"exp_{data_rows + 1:04d}"


def currentBestScore(results_path: Path, primary_metric: str) -> Optional[float]:
    if not results_path.exists():
        return None
    best: Optional[float] = None
    with results_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("primary_metric") != primary_metric:
                continue
            raw_score = row.get("primary_score", "")
            if not raw_score:
                continue
            score = float(raw_score)
            if best is None or score > best:
                best = score
    return best


def formatMetric(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def appendResultsRow(results_path: Path, row: dict[str, Any]) -> None:
    ensureDir(results_path.parent)
    write_header = not results_path.exists()
    with results_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({key: formatMetric(row.get(key, "")) for key in RESULTS_HEADER})


def serializableCandidate(name: str, notes: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "notes": notes,
        "estimator_class": result["estimator_class"],
        "estimator_params": result["estimator_params"],
        "validation_metrics": result["validation_metrics"],
        "train_metrics": result["train_metrics"],
        "cv_summary": result["cv_summary"],
        "cv_rows": result["cv_rows"],
        "fit_seconds": result["fit_seconds"],
        "cv_fit_seconds": result["cv_fit_seconds"],
        "generalization_gap": result["generalization_gap"],
        "decision_threshold": result["decision_threshold"],
        "train_feature_count": result["train_feature_count"],
        "training_history": result.get("training_history"),
    }


def saveBestModel(model: Pipeline, path: Path) -> None:
    try:
        import joblib

        ensureDir(path.parent)
        joblib.dump(model, path)
    except Exception:
        return


def buildValidationPredictions(
    prepared: PreparedExperiment,
    model: Pipeline,
    decisionThreshold: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    predictions, probabilities, _ = predictModelOutputs(model, prepared.X_val)
    if decisionThreshold is not None and probabilities is not None:
        positive_scores = probabilityScores(probabilities)
        model_classes = getattr(model.named_steps["model"], "classes_", None)
        if positive_scores is not None and model_classes is not None and len(model_classes) == 2:
            predictions = thresholdPredictions(model_classes, positive_scores, decisionThreshold["threshold"])

    frame = pd.DataFrame(
        {
            "actual": prepared.y_val.reset_index(drop=True),
            "prediction": pd.Series(predictions).reset_index(drop=True),
        }
    )
    if probabilities is not None:
        model_step = model.named_steps["model"]
        classes = getattr(model_step, "classes_", None)
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            frame["prob_positive"] = probabilities[:, 1]
        elif probabilities.ndim == 2 and classes is not None:
            for index, label in enumerate(classes):
                frame[f"prob_{label}"] = probabilities[:, index]
    if decisionThreshold is not None:
        frame["decision_threshold"] = decisionThreshold["threshold"]
        frame["threshold_metric"] = decisionThreshold["metric"]
    return frame


def saveValidationPredictions(
    prepared: PreparedExperiment,
    model: Pipeline,
    path: Path,
    decisionThreshold: Optional[dict[str, Any]] = None,
) -> None:
    ensureDir(path.parent)
    buildValidationPredictions(prepared, model, decisionThreshold=decisionThreshold).to_csv(path, index=False)


def configToDict(config: ExperimentConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config)))
