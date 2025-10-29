import json
from typing import Any, Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import manifold
from sklearn.model_selection import train_test_split

sns.set(rc={"figure.figsize": (11.7, 8.27)})


def _build_palette(labels: np.ndarray):
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", max(len(unique_labels), 1))
    color_map = {label: palette[idx % len(palette)] for idx, label in enumerate(unique_labels)}
    return color_map, list(unique_labels)


def _resolve_label_name(label: Any, label_map: Mapping[int, str] | None) -> str:
    if label_map is None:
        return str(label)
    try:
        key = int(label)
    except (TypeError, ValueError):
        return str(label)
    return label_map.get(key, str(label))


def _prepare_label_names(
    labels: Iterable[Any], label_map: Mapping[int, str] | None
) -> np.ndarray:
    return np.asarray([_resolve_label_name(label, label_map) for label in labels])


def plot_embedding(X_org, y, title=None, color_labels=None, label_map: Mapping[int, str] | None = None):
    """Project *X_org* with t-SNE and colour the scatter plot by labels."""

    if color_labels is not None:
        X, _, Y, _, display_labels, _ = train_test_split(X_org, y, color_labels, test_size=0.5)
        display_labels = np.asarray(display_labels)
    else:
        X, _, Y, _ = train_test_split(X_org, y, test_size=0.5)
        display_labels = np.asarray(Y)
    X, Y = np.asarray(X), np.asarray(Y)

    tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
    print("Fitting TSNE!")
    X = tsne.fit_transform(X)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    coordinates = X.tolist() if isinstance(X, np.ndarray) else X
    binary_labels = Y.tolist() if isinstance(Y, np.ndarray) else Y
    colour_payload = display_labels.tolist()

    label_names_payload = _prepare_label_names(display_labels, label_map).tolist()

    with open(str(title) + "-tsne-features.json", "w") as handle:
        json.dump([coordinates, colour_payload], handle)

    if color_labels is not None:
        metadata_path = str(title) + "-tsne-metadata.json"
        with open(metadata_path, "w") as meta_handle:
            metadata: Dict[str, Any] = {
                "binary_labels": binary_labels,
                "color_labels": colour_payload,
            }
            if label_map is not None:
                metadata["color_label_names"] = label_names_payload
            json.dump(metadata, meta_handle)
    elif label_map is not None:
        metadata_path = str(title) + "-tsne-metadata.json"
        with open(metadata_path, "w") as meta_handle:
            json.dump({"color_label_names": label_names_payload}, meta_handle)

    fig, ax = plt.subplots()
    color_map, ordered_labels = _build_palette(display_labels)
    colors = [color_map[label] for label in display_labels]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=18, linewidths=0)

    ordered_label_names = _prepare_label_names(ordered_labels, label_map)

    if len(ordered_labels) <= 20:
        legend_handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=color_map[label],
                       markeredgecolor="black", label=ordered_label_names[idx])
            for idx, label in enumerate(ordered_labels)
        ]
        ax.legend(handles=legend_handles, title="Label", loc="best", frameon=True)

    if X.shape[0] <= 400:
        for x_coord, y_coord, label_name in zip(X[:, 0], X[:, 1], label_names_payload):
            ax.text(
                x_coord,
                y_coord,
                label_name,
                fontsize=6,
                alpha=0.8,
                ha="left",
                va="bottom",
            )

    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title("")
    fig.tight_layout()
    fig.savefig(str(title) + ".pdf")
    plt.show()


if __name__ == '__main__':
    x_a = np.random.uniform(0, 1, size=(32, 256))
    targets = np.random.randint(0, 2, size=(32))
    print(targets)
    plot_embedding(x_a, targets)
    print("Computing t-SNE embedding")