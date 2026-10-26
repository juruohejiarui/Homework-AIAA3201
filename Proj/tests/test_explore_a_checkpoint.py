"""Checkpoint recovery must reproduce the next optimizer update and sampling RNG."""
import random

import numpy as np
import torch

from explore_a.training.checkpoint import load_checkpoint, save_checkpoint, seed_everything


def test_resume_restores_optimizer_and_random_streams(tmp_path):
    seed_everything(2026)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    inputs = torch.ones(2, 3)
    def update():
        optimizer.zero_grad()
        model(inputs).square().mean().backward()
        optimizer.step()
        scheduler.step()
    update()
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, scheduler, scaler, 1, .2, {"experiment": "test"})
    expected_rng = (random.random(), np.random.rand(), torch.rand(1))
    update()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    state = load_checkpoint(path, model, optimizer, scheduler, scaler, {"experiment": "test"})
    assert state["step"] == 1
    assert random.random() == expected_rng[0]
    assert np.random.rand() == expected_rng[1]
    torch.testing.assert_close(torch.rand(1), expected_rng[2])
    update()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
