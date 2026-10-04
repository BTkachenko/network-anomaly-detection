from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from clustering_stage import distance_to_centers, evaluate_birch
from data_utils import canonical_header, metrics_table, prepare_features, sample_summary
from simple_iforest import SimpleIsolationForest


class PreprocessingTests(unittest.TestCase):
    def test_training_statistics_are_not_affected_by_evaluation_data(self):
        train = pd.DataFrame(
            {
                "bytes": [1.0, 3.0, np.nan],
                "constant": [7, 7, 7],
                "Label": [0, 0, 0],
                "Src IP": ["a", "b", "c"],
                "source_row": [1, 2, 3],
                "binary_label": [0, 0, 0],
            }
        )
        evaluation = pd.DataFrame(
            {
                "bytes": [np.nan, np.inf, 100.0],
                "constant": [9, 9, 9],
                "Label": [1, 1, 1],
                "Src IP": ["x", "y", "z"],
                "source_row": [4, 5, 6],
                "binary_label": [1, 1, 1],
            }
        )
        x_train, x_eval, columns = prepare_features(train, evaluation)
        self.assertEqual(columns, ["bytes"])
        np.testing.assert_allclose(x_train[:, 0], [-np.sqrt(1.5), np.sqrt(1.5), 0])
        np.testing.assert_allclose(x_eval[:2, 0], [0, 0])
        self.assertGreater(x_eval[2, 0], 100)
        self.assertTrue(np.isfinite(x_train).all() and np.isfinite(x_eval).all())

    def test_missing_dataset_directory_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "NETWORK_ANOMALY_DATA_DIR"):
                canonical_header(Path(directory) / "missing")

    def test_sample_summary_reports_record_overlap(self):
        train = pd.DataFrame({"source_file": ["a.csv"], "source_row": [1], "binary_label": [0]})
        evaluation = pd.DataFrame(
            {
                "source_file": ["a.csv", "b.csv"],
                "source_row": [1, 2],
                "binary_label": [0, 1],
                "coarse_label": ["Benign", "DoS"],
            }
        )
        summary = sample_summary(train, evaluation).set_index("item")["value"]
        self.assertEqual(summary["train_eval_benign_overlap"], 1)


class MetricTests(unittest.TestCase):
    def test_confusion_counts_and_false_positive_rate(self):
        frame = pd.DataFrame({"binary_label": [0, 0, 1, 1]})
        result = metrics_table(frame, np.array([0, 1, 1, 0]))
        self.assertEqual([result[k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        self.assertEqual(result["fpr"], 0.5)
        self.assertEqual(result["f1"], 0.5)

    def test_no_positive_prediction_returns_zero_precision_and_recall(self):
        result = metrics_table(pd.DataFrame({"binary_label": [0, 1]}), np.array([0, 0]))
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["fpr"], 0)


class ModelTests(unittest.TestCase):
    def test_birch_scores_a_distant_point_as_anomaly(self):
        train = np.array([[0.0], [0.1], [0.2], [0.3]])
        evaluation = np.array([[0.15], [10.0]])
        frame = pd.DataFrame({"binary_label": [0, 1]})
        metrics, predictions = evaluate_birch(
            train,
            evaluation,
            frame,
            {
                "name": "test",
                "threshold": 0.5,
                "branching_factor": 10,
                "score_percentile": 95,
            },
        )
        np.testing.assert_array_equal(predictions, [0, 1])
        self.assertEqual(metrics["f1"], 1)

    def test_empty_cluster_representatives_have_infinite_distance(self):
        distances = distance_to_centers(np.array([[1.0], [2.0]]), np.empty((0, 1)))
        self.assertTrue(np.isinf(distances).all())

    def test_custom_isolation_forest_is_repeatable_for_a_fixed_seed(self):
        train = np.random.default_rng(17).normal(size=(40, 3))
        first = SimpleIsolationForest(n_trees=8, sample_size=16, random_state=21).fit(train)
        second = SimpleIsolationForest(n_trees=8, sample_size=16, random_state=21).fit(train)
        np.testing.assert_array_equal(first.score_samples(train), second.score_samples(train))

    def test_custom_isolation_forest_requires_fit(self):
        with self.assertRaises(RuntimeError):
            SimpleIsolationForest().score_samples(np.array([[1.0]]))


if __name__ == "__main__":
    unittest.main()
