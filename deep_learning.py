
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


def isTorchAvailable() -> bool:
    return TORCH_AVAILABLE


def ensureTorch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed. Install torch to enable deep learning candidates.")


def resolveDevice(device: str) -> str:
    ensureTorch()
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _TabularMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class _TorchTabularBase(BaseEstimator):
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 30,
        patience: int = 6,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def seedAll(self) -> None:
        ensureTorch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def prepareFeatures(self, X: Any) -> np.ndarray:
        features = np.asarray(X, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        return features

    def splitTrainingData(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.validation_fraction <= 0 or len(X) < 20:
            return X, X, y, y
        return train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=stratify,
        )

    def buildLoader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        ensureTorch()
        feature_tensor = torch.tensor(X, dtype=torch.float32)
        target_tensor = self.targetTensor(y)
        dataset = TensorDataset(feature_tensor, target_tensor)
        return DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=shuffle)

    def fitNetwork(
        self,
        network: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
    ) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = self.buildLoss()

        best_loss = float("inf")
        best_epoch = 0
        best_state = deepcopy(network.state_dict())
        patience_counter = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, self.max_epochs + 1):
            network.train()
            running_train_loss = 0.0
            train_examples = 0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad()
                outputs = network(batch_features)
                loss = self.lossFromOutputs(criterion, outputs, batch_targets)
                loss.backward()
                optimizer.step()
                running_train_loss += float(loss.item()) * len(batch_features)
                train_examples += len(batch_features)

            network.eval()
            running_val_loss = 0.0
            val_examples = 0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    outputs = network(batch_features)
                    loss = self.lossFromOutputs(criterion, outputs, batch_targets)
                    running_val_loss += float(loss.item()) * len(batch_features)
                    val_examples += len(batch_features)

            train_loss = running_train_loss / max(1, train_examples)
            val_loss = running_val_loss / max(1, val_examples)
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(network.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose:
                print(f"[torch] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if patience_counter >= self.patience:
                break

        network.load_state_dict(best_state)
        return {
            "epochs": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "device": device,
        }

    def predictBatches(self, X: Any) -> np.ndarray:
        ensureTorch()
        features = self.prepareFeatures(X)
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        loader = DataLoader(TensorDataset(feature_tensor), batch_size=min(self.batch_size, len(features)), shuffle=False)

        self.network_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for (batch_features,) in loader:
                logits = self.network_(batch_features.to(self.device_))
                outputs.append(logits.detach().cpu().numpy())
        return np.vstack(outputs)


class TorchTabularClassifier(_TorchTabularBase, ClassifierMixin):
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 30,
        patience: int = 6,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "auto",
        class_weight: str | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            hidden_dims=hidden_dims,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            validation_fraction=validation_fraction,
            random_state=random_state,
            device=device,
            verbose=verbose,
        )
        self.class_weight = class_weight

    def targetTensor(self, y: np.ndarray) -> torch.Tensor:
        if self.n_classes_ == 2:
            return torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        return torch.tensor(y, dtype=torch.long)

    def buildLoss(self) -> Any:
        if self.n_classes_ == 2:
            if self.class_weight == "balanced":
                positives = max(1, int(np.sum(self.y_train_encoded_ == 1)))
                negatives = max(1, int(np.sum(self.y_train_encoded_ == 0)))
                pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=self.device_)
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            return nn.BCEWithLogitsLoss()

        if self.class_weight == "balanced":
            class_counts = np.bincount(self.y_train_encoded_, minlength=self.n_classes_).astype(np.float32)
            weights = class_counts.sum() / np.maximum(1.0, class_counts)
            weights = weights / weights.mean()
            return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=self.device_))
        return nn.CrossEntropyLoss()

    def lossFromOutputs(self, criterion: Any, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.n_classes_ == 2:
            return criterion(outputs, targets)
        return criterion(outputs, targets)

    def fit(self, X: Any, y: Any) -> "TorchTabularClassifier":
        ensureTorch()
        self.seedAll()
        features = self.prepareFeatures(X)
        y_array = np.asarray(y)
        self.classes_, encoded = np.unique(y_array, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.y_train_encoded_ = encoded
        self.device_ = resolveDevice(self.device)

        stratify = encoded if self.n_classes_ > 1 else None
        X_train, X_val, y_train, y_val = self.splitTrainingData(features, encoded, stratify=stratify)
        self.y_train_encoded_ = y_train

        output_dim = 1 if self.n_classes_ == 2 else self.n_classes_
        self.network_ = _TabularMLP(
            input_dim=features.shape[1],
            output_dim=output_dim,
            hidden_dims=tuple(self.hidden_dims),
            dropout=self.dropout,
        ).to(self.device_)

        train_loader = self.buildLoader(X_train, y_train, shuffle=True)
        val_loader = self.buildLoader(X_val, y_val, shuffle=False)
        history = self.fitNetwork(self.network_, train_loader, val_loader, self.device_)
        self.training_history_ = {
            "type": "torch_mlp_classifier",
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "validation_fraction": self.validation_fraction,
            **history,
        }
        return self

    def predictProba(self, X: Any) -> np.ndarray:
        logits = self.predictBatches(X)
        if self.n_classes_ == 2:
            probabilities = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
            return np.column_stack([1.0 - probabilities, probabilities])
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predictProba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def decisionFunction(self, X: Any) -> np.ndarray:
        logits = self.predictBatches(X)
        if self.n_classes_ == 2:
            return logits.reshape(-1)
        return logits

    predict_proba = predictProba
    decision_function = decisionFunction


class TorchTabularRegressor(_TorchTabularBase, RegressorMixin):
    def targetTensor(self, y: np.ndarray) -> torch.Tensor:
        return torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    def buildLoss(self) -> Any:
        return nn.MSELoss()

    def lossFromOutputs(self, criterion: Any, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return criterion(outputs, targets)

    def fit(self, X: Any, y: Any) -> "TorchTabularRegressor":
        ensureTorch()
        self.seedAll()
        features = self.prepareFeatures(X)
        targets = np.asarray(y, dtype=np.float32)
        self.device_ = resolveDevice(self.device)

        X_train, X_val, y_train, y_val = self.splitTrainingData(features, targets, stratify=None)
        self.network_ = _TabularMLP(
            input_dim=features.shape[1],
            output_dim=1,
            hidden_dims=tuple(self.hidden_dims),
            dropout=self.dropout,
        ).to(self.device_)

        train_loader = self.buildLoader(X_train, y_train, shuffle=True)
        val_loader = self.buildLoader(X_val, y_val, shuffle=False)
        history = self.fitNetwork(self.network_, train_loader, val_loader, self.device_)
        self.training_history_ = {
            "type": "torch_mlp_regressor",
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "validation_fraction": self.validation_fraction,
            **history,
        }
        return self

    def predict(self, X: Any) -> np.ndarray:
        outputs = self.predictBatches(X)
        return outputs.reshape(-1)
