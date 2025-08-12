from inference.predict_utils import (
    parse_comma_separated_floats,
    average_scalar_probabilities,
    apply_temperature_to_prob,
)


def test_parse_and_average_probs():
    weights = parse_comma_separated_floats("0.2,0.3,0.5")
    assert len(weights) == 3
    avg = average_scalar_probabilities([0.2, 0.4, 0.8], weights)
    assert 0.0 <= avg <= 1.0


def test_temperature_application():
    p = 0.8
    p_cool = apply_temperature_to_prob(p, temperature=2.0)  # should be less confident
    p_hot = apply_temperature_to_prob(p, temperature=0.5)   # should be more confident
    assert 0.5 < p_cool < p < p_hot < 1.0


def test_parse_none_and_empty():
    assert parse_comma_separated_floats(None) is None
    assert parse_comma_separated_floats("") is None


def test_average_probs_unweighted_and_weighted_precise():
    probs = [0.2, 0.4, 0.8]
    # unweighted mean
    avg_u = average_scalar_probabilities(probs)
    assert abs(avg_u - (sum(probs) / len(probs))) < 1e-8
    # weighted
    weights = [0.2, 0.3, 0.5]
    avg_w = average_scalar_probabilities(probs, weights)
    expected = 0.2 * 0.2 + 0.3 * 0.4 + 0.5 * 0.8
    assert abs(avg_w - expected) < 1e-8


def test_average_probs_mismatch_raises():
    try:
        average_scalar_probabilities([0.1, 0.9], [1.0])
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError:
        pass


def test_average_probs_zero_weight_sum_raises():
    try:
        average_scalar_probabilities([0.1, 0.9], [0.0, 0.0])
        assert False, "Expected ValueError for zero weights sum"
    except ValueError:
        pass


def test_temperature_behavior_below_and_above_half():
    # For p < 0.5, cooling (T>1) increases toward 0.5, heating (T<1) decreases away
    p = 0.2
    p_cool = apply_temperature_to_prob(p, 2.0)
    p_hot = apply_temperature_to_prob(p, 0.5)
    assert 0.2 < p_cool < 0.5
    assert 0.0 < p_hot < 0.2


