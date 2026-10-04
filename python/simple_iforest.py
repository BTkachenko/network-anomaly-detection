from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import numpy as np


@dataclass
class IsolationNode:
    """Single node in a simple Isolation Forest tree."""

    size: int
    feature: int | None = None
    threshold: float | None = None
    left: "IsolationNode | None" = None
    right: "IsolationNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def average_path_length(sample_size: int) -> float:
    """Expected path length of an unsuccessful search in a binary tree."""
    if sample_size <= 1:
        return 0.0
    if sample_size == 2:
        return 1.0
    harmonic = np.sum(1.0 / np.arange(1, sample_size))
    return 2.0 * harmonic - (2.0 * (sample_size - 1) / sample_size)


class SimpleIsolationForest:
    """
    Small teaching-oriented Isolation Forest implementation.

    It is intentionally compact so the Python and Julia versions stay easy
    to compare. The code is enough for the current project sample, but it is
    not meant to replace a production library.
    """

    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 256,
        max_depth: int | None = None,
        random_state: int = 2029,
    ) -> None:
        self.n_trees = int(n_trees)
        self.sample_size = int(sample_size)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.trees_: list[IsolationNode] = []
        self.sample_size_: int = 0
        self.max_depth_: int = 0

    def fit(self, x: np.ndarray) -> "SimpleIsolationForest":
        rng = np.random.default_rng(self.random_state)
        self.sample_size_ = min(self.sample_size, len(x))
        self.max_depth_ = self.max_depth or int(ceil(log2(max(self.sample_size_, 2))))
        self.trees_ = []

        for _ in range(self.n_trees):
            indices = rng.choice(len(x), size=self.sample_size_, replace=False)
            sample = x[indices]
            self.trees_.append(self._build_tree(sample, depth=0, rng=rng))

        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Model musi zostac wytrenowany przed wywolaniem score_samples().")

        scores = np.zeros(len(x), dtype=float)
        normalizer = average_path_length(self.sample_size_)
        if normalizer == 0.0:
            return scores

        for tree in self.trees_:
            self._accumulate_path_lengths(x, tree, np.arange(len(x)), depth=0, out=scores)

        average_depth = scores / self.n_trees
        return np.power(2.0, -average_depth / normalizer)

    def _build_tree(self, x: np.ndarray, depth: int, rng: np.random.Generator) -> IsolationNode:
        if depth >= self.max_depth_ or len(x) <= 1:
            return IsolationNode(size=len(x))

        mins = np.min(x, axis=0)
        maxs = np.max(x, axis=0)
        valid_features = np.flatnonzero(maxs > mins)
        if len(valid_features) == 0:
            return IsolationNode(size=len(x))

        feature = int(rng.choice(valid_features))
        threshold = float(rng.uniform(mins[feature], maxs[feature]))
        left_mask = x[:, feature] < threshold
        right_mask = ~left_mask

        if not left_mask.any() or not right_mask.any():
            return IsolationNode(size=len(x))

        return IsolationNode(
            size=len(x),
            feature=feature,
            threshold=threshold,
            left=self._build_tree(x[left_mask], depth + 1, rng),
            right=self._build_tree(x[right_mask], depth + 1, rng),
        )

    def _path_length(self, row: np.ndarray, node: IsolationNode, depth: int) -> float:
        if node.is_leaf:
            return depth + average_path_length(node.size)

        assert node.feature is not None
        assert node.left is not None
        assert node.right is not None
        assert node.threshold is not None

        if row[node.feature] < node.threshold:
            return self._path_length(row, node.left, depth + 1)
        return self._path_length(row, node.right, depth + 1)

    def _accumulate_path_lengths(
        self,
        x: np.ndarray,
        node: IsolationNode,
        indices: np.ndarray,
        depth: int,
        out: np.ndarray,
    ) -> None:
        """Accumulate path lengths for a batch of rows at once."""
        if indices.size == 0:
            return

        if node.is_leaf:
            out[indices] += depth + average_path_length(node.size)
            return

        assert node.feature is not None
        assert node.left is not None
        assert node.right is not None
        assert node.threshold is not None

        left_mask = x[indices, node.feature] < node.threshold
        self._accumulate_path_lengths(x, node.left, indices[left_mask], depth + 1, out)
        self._accumulate_path_lengths(x, node.right, indices[~left_mask], depth + 1, out)
