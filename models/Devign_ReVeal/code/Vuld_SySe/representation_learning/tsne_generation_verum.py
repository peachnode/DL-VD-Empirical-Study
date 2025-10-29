"""Generate t-SNE visualisations for GGNN and ReVeal representations.

This script is a CLI-friendly rewrite of the original helper that shipped with
ReVeal.  Instead of editing constants inside the module you can now point it at
any processed dataset directory (the same one consumed by ``api_test.py``) and
request plots for the chosen split with a couple of command-line flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from Vuld_SySe.representation_learning.representation_learning_api import (
    RepresentationLearningModel,
)
from Vuld_SySe.representation_learning.trainer import show_representation
from Vuld_SySe.representation_learning.tsne import plot_embedding

LABEL_MAP: Dict[str, int] = {
    "NOT_HELPFUL": 0,
    "SATURATED": 1,
    "UNREACHED": 2,
    "BUILD_ERROR": 3,
    "SUCCESS": 4,
    "WRONG_FORMAT": 5,
    "INSERT_ERROR": 6,
}

LABEL_NAME_BY_ID: Mapping[int, str] = {value: key for key, value in LABEL_MAP.items()}


def parse_vulnerable_labels(values: Sequence[str] | None) -> List[int]:
    """Return a sorted list of labels that should be treated as vulnerable."""

    if not values:
        return [1]

    labels = set()
    for value in values:
        for chunk in value.split(","):
            part = chunk.strip()
            if not part:
                continue
            try:
                labels.add(int(part))
            except ValueError as exc:  # pragma: no cover - defensive programming
                raise argparse.ArgumentTypeError(f"Invalid label '{part}'") from exc

    if not labels:
        raise argparse.ArgumentTypeError("At least one vulnerable label must be provided")

    return sorted(labels)


def load_records(path: Path, *, rng: np.random.Generator, shuffle: bool = False) -> List[Dict]:
    """Load the JSON or JSONL records stored at *path* and optionally shuffle them."""

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".jsonlines", ".jsonl"}:
        import jsonlines

        with jsonlines.open(path) as reader:  # type: ignore[assignment]
            records = list(reader)
    else:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

    if shuffle:
        rng.shuffle(records)
    return records


def read_split(path: Path, *, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return feature, target, and raw-label arrays loaded from *path*."""

    features: List[Sequence[float]] = []
    targets: List[int] = []
    raw_targets: List[int] = []

    for entry in load_records(path, rng=rng, shuffle=False):
        try:
            features.append(entry["graph_feature"])
            targets.append(int(entry["target"]))
            raw_targets.append(int(entry.get("raw_target", entry["target"])))
        except KeyError as exc:
            raise KeyError(f"Missing key {exc!s} in record from {path}") from exc

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(raw_targets, dtype=np.int64),
    )


def _coerce_label(value) -> int:
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Encountered an empty label sequence while extracting raw targets")
        value = value[0]
    if isinstance(value, (np.integer, int, bool)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Encountered an empty label string while extracting raw targets")
        try:
            return int(stripped)
        except ValueError:
            return int(float(stripped))
    raise TypeError(f"Unsupported label type: {type(value)!r}")


def _extract_raw_label(entry: Dict) -> int:
    if "raw_target" in entry:
        return _coerce_label(entry["raw_target"])
    if "label" in entry:
        return _coerce_label(entry["label"])
    if "targets" in entry:
        return _coerce_label(entry["targets"])
    if "target" in entry:
        return _coerce_label(entry["target"])
    raise KeyError("Record does not contain a recognised label field")


def load_full_graph_labels(dataset_dir: Path, *, rng: np.random.Generator) -> np.ndarray | None:
    """Load raw labels from the cached full-graph dataset if available."""

    dataset_root = dataset_dir.parent
    dataset_name = dataset_root.parent.name if dataset_root.parent != dataset_root else dataset_root.name
    candidates = [
        dataset_root / f"{dataset_name}-full_graph.jsonlines",
        dataset_root / f"{dataset_name}-full_graph.jsonl",
        dataset_root / f"{dataset_name}-full_graph.json",
        dataset_root / "full_graph.jsonlines",
        dataset_root / "full_graph.jsonl",
        dataset_root / "full_graph.json",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        records = load_records(candidate, rng=rng, shuffle=False)
        try:
            labels = [_extract_raw_label(record) for record in records]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Failed to extract raw labels from {candidate}") from exc
        return np.asarray(labels, dtype=np.int64)

    return None


def load_dataset(
    dataset_dir: Path,
    *,
    seed: int,
    vulnerable_labels: Iterable[int],
    features_suffix: str | None,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load the consolidated GGNN features and return split views."""

    if features_suffix:
        features_file = dataset_dir / f"GGNNinput_graph_ggnn_{seed}_{features_suffix}.jsonlines"
    else:
        features_file = dataset_dir / f"GGNNinput_graph_ggnn_{seed}.jsonlines"

    features, stored_targets, raw_targets = read_split(features_file, rng=rng)

    full_graph_raw = load_full_graph_labels(dataset_dir, rng=rng)
    if full_graph_raw is not None:
        if full_graph_raw.shape[0] != features.shape[0]:
            raise ValueError(
                "Mismatch between GGNN feature count and full-graph label count: "
                f"{features.shape[0]} features vs {full_graph_raw.shape[0]} labels."
            )
        raw_targets = full_graph_raw

    vulnerable = set(vulnerable_labels)
    targets = np.asarray([1 if int(label) in vulnerable else 0 for label in stored_targets], dtype=np.int64)

    with (dataset_dir / "splits_reveal.json").open("r", encoding="utf-8") as handle:
        raw_splits = json.load(handle)

    indices: Dict[str, List[int]] = {"train": [], "valid": [], "test": [], "holdout": []}
    for index, split_name in raw_splits.items():
        try:
            indices[split_name].append(int(index))
        except KeyError as exc:
            raise KeyError(f"Unknown split '{split_name}' in splits_reveal.json") from exc

    sliced_indices = {name: np.asarray(idxs, dtype=np.int64) for name, idxs in indices.items()}
    return features, targets, raw_targets, sliced_indices


def ensure_output(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory produced by full_data_prep_script.py (e.g. data/<ds>/full_experiment_real_data_processed)",
    )
    parser.add_argument("--fold", required=True, help="Fold identifier used during training (e.g. 1)")
    parser.add_argument("--seed", required=True, type=int, help="Random seed used when generating GGNN features")
    parser.add_argument(
        "--output-dir",
        default="tsnes",
        help="Where to store the generated <name>.pdf and <name>-tsne-features.json files",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test", "holdout"),
        default="test",
        help="Dataset split to visualise",
    )
    parser.add_argument(
        "--features-suffix",
        default=None,
        help="Optional suffix appended to GGNNinput_graph_ggnn_<seed>_<suffix>.jsonlines",
    )
    parser.add_argument(
        "--vulnerable-labels",
        nargs="*",
        default=None,
        help="Labels that should be treated as vulnerable (comma- or space-separated integers)",
    )
    parser.add_argument("--num-epoch", default=100, type=int, help="Number of training epochs for the representation model")
    parser.add_argument("--max-patience", default=5, type=int, help="Early stopping patience for the representation model")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size for the representation model")
    parser.add_argument("--hidden-dim", default=256, type=int, help="Hidden dimension of the metric-learning head")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability for the metric-learning head")
    parser.add_argument("--alpha", default=0.5, type=float, help="Alpha hyper-parameter for the metric-learning loss")
    parser.add_argument("--lambda1", default=0.5, type=float, help="Lambda1 hyper-parameter for the metric-learning loss")
    parser.add_argument("--lambda2", default=0.001, type=float, help="Lambda2 hyper-parameter for the metric-learning loss")
    parser.add_argument("--num-layers", default=1, type=int, help="Number of layers in the metric-learning head")
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable SMOTE balancing inside the representation-learning dataset",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Where to store the intermediate RepresentationLearningModel checkpoints (defaults to models/devign/reveal/v<fold>/<seed>)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Seed used when shuffling records before plotting",
    )

    args = parser.parse_args()

    try:
        vulnerable_labels = parse_vulnerable_labels(args.vulnerable_labels)
    except argparse.ArgumentTypeError as exc:  # pragma: no cover - arg parsing
        parser.error(str(exc))

    rng = np.random.default_rng(args.rng_seed)

    dataset_root = Path(args.input_dir)
    dataset_dir = dataset_root / f"v{args.fold}"
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    features, targets, raw_targets, split_indices = load_dataset(
        dataset_dir,
        seed=args.seed,
        vulnerable_labels=vulnerable_labels,
        features_suffix=args.features_suffix,
        rng=rng,
    )

    if args.split not in split_indices:
        raise KeyError(f"Unknown split '{args.split}'. Available keys: {sorted(split_indices)}")

    if split_indices[args.split].size == 0:
        raise ValueError(f"Split '{args.split}' is empty; cannot generate a t-SNE plot")

    selected_indices = split_indices[args.split]
    selected_features = features[selected_indices]
    selected_targets = targets[selected_indices]
    selected_raw_targets = raw_targets[selected_indices]
    selected_raw_targets_list = selected_raw_targets.tolist()

    output_dir = Path(args.output_dir)
    ggnn_title = output_dir / f"{args.split}-ggnn"
    ensure_output(ggnn_title)
    print(f"Saving GGNN t-SNE plot to {ggnn_title}")
    plot_embedding(
        selected_features,
        selected_targets,
        str(ggnn_title),
        color_labels=selected_raw_targets,
        label_map=LABEL_NAME_BY_ID,
    )

    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else Path("models") / "devign" / "reveal" / dataset_dir.name / str(args.seed)
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    train_idx = split_indices["train"]
    valid_idx = split_indices["valid"]

    if train_idx.size == 0 or valid_idx.size == 0:
        raise ValueError("Both the train and valid splits must be non-empty to train the representation model")

    model = RepresentationLearningModel(
        alpha=args.alpha,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        balance=not args.no_balance,
        num_epoch=args.num_epoch,
        max_patience=args.max_patience,
        num_layers=args.num_layers,
    )

    model.train(
        features[train_idx],
        targets[train_idx],
        features[valid_idx],
        targets[valid_idx],
        str(model_dir),
    )

    # Prepare the chosen split for t-SNE in the learned representation space.
    model.dataset.clear_test_set()
    for feature, label in zip(selected_features, selected_targets):
        model.dataset.add_data_entry(feature.tolist(), int(label), part="test")

    cuda_device = 0 if model.cuda else -1
    representation_title = output_dir / f"{args.split}-representation"
    ensure_output(representation_title)
    print(f"Saving representation t-SNE plot to {representation_title}")
    previous_shuffle = model.dataset.shuffle
    model.dataset.shuffle = False
    try:
        show_representation(
            model.model,
            model.dataset.get_next_test_batch,
            model.dataset.initialize_test_batches(),
            cuda_device,
            str(representation_title),
            color_labels=selected_raw_targets_list,
            label_map=LABEL_NAME_BY_ID,
        )
    finally:
        model.dataset.shuffle = previous_shuffle


if __name__ == "__main__":
    main()

