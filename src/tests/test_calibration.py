import torch

from utils.calibration import TemperatureScaler


def test_temperature_scaler_fit_and_apply():
    # Synthetic binary probs with slight miscalibration
    torch.manual_seed(0)
    targets = torch.randint(low=0, high=2, size=(200,), dtype=torch.int64)
    logits = torch.randn(200)
    probs = torch.sigmoid(logits * 1.5)  # overconfident

    scaler = TemperatureScaler(init_temperature=1.0)
    T_before = float(scaler.temperature().item())

    T_after = scaler.fit_from_probs(probs, targets)

    assert T_after > 0
    # sanity: temperature changed
    assert abs(T_after - T_before) > 1e-4

    # Applying temperature should change probabilities
    probs_cal = scaler.forward_probs(probs)
    assert torch.mean(torch.abs(probs_cal - probs)) > 1e-4

    # Check Brier computation runs
    _b_before = TemperatureScaler.brier_score(probs, targets.float())
    _b_after = TemperatureScaler.brier_score(probs_cal, targets.float())


