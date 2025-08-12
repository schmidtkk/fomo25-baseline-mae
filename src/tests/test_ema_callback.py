import torch

from utils.callbacks import ModelEMACallback


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2, bias=False)


def test_ema_updates_and_swaps():
    m = DummyModule()
    cb = ModelEMACallback(decay=0.5)

    class DummyTrainer:
        pass

    trainer = DummyTrainer()

    # Initialize EMA with current weights
    cb.on_fit_start(trainer, m)
    ema_before = {k: v.clone() for k, v in cb._ema_state.items()}

    # Make an SGD-like update to model params and call on_train_batch_end
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    cb.on_train_batch_end(trainer, m, outputs=None, batch=None, batch_idx=0)

    # EMA should move towards new params; not equal to old EMA and not equal to new params exactly
    for name, p in m.state_dict().items():
        if name in cb._ema_state and p.is_floating_point():
            ema_val = cb._ema_state[name]
            assert not torch.allclose(ema_val, ema_before[name])
            assert not torch.allclose(ema_val, p)

    # Swap into validation and restore back
    state_before = {k: v.clone() for k, v in m.state_dict().items()}
    cb.on_validation_start(trainer, m)
    # After swap, model state should match EMA (for float tensors tracked by EMA)
    for name, p in m.state_dict().items():
        if name in cb._ema_state and p.is_floating_point():
            assert torch.allclose(p, cb._ema_state[name].to(p.device))
    cb.on_validation_end(trainer, m)
    # Restored
    for name, p in m.state_dict().items():
        assert torch.allclose(p, state_before[name])


