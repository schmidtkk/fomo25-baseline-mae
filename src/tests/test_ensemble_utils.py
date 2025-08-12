import numpy as np

from utils.ensemble import (
    average_probabilities,
    average_subject_probabilities,
    auroc_from_subject_probabilities,
)


def test_average_probabilities_unweighted_and_weighted():
    p1 = np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
    p2 = np.array([[0.4, 0.6], [0.5, 0.5]], dtype=np.float32)

    avg = average_probabilities([p1, p2])
    expected = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32)
    assert np.allclose(avg, expected, atol=1e-6)

    wavg = average_probabilities([p1, p2], weights=[0.25, 0.75])
    expected_w = 0.25 * p1 + 0.75 * p2
    assert np.allclose(wavg, expected_w, atol=1e-6)


def test_average_subject_probabilities_and_auroc():
    # Two folds with identical subject sets
    fold1 = {"A": 0.8, "B": 0.3, "C": 0.6}
    fold2 = {"A": 0.7, "B": 0.5, "C": 0.2}

    avg = average_subject_probabilities([fold1, fold2])
    assert set(avg.keys()) == {"A", "B", "C"}
    assert np.isclose(avg["A"], 0.75, atol=1e-6)
    assert np.isclose(avg["B"], 0.4, atol=1e-6)
    assert np.isclose(avg["C"], 0.4, atol=1e-6)

    targets = {"A": 1, "B": 0, "C": 1}
    auroc_val = auroc_from_subject_probabilities(avg, targets)
    assert np.isfinite(auroc_val)


def test_average_subject_probabilities_requires_identical_keys():
    fold1 = {"A": 0.8, "B": 0.3}
    fold2 = {"A": 0.7, "C": 0.5}
    try:
        average_subject_probabilities([fold1, fold2])
        assert False, "Expected ValueError for mismatched subject keys"
    except ValueError:
        pass


