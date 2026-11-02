"""Checkpoint recovery must reproduce the next optimizer update and sampling RNG."""
import random

import numpy as np
import torch

from explore_a.training.checkpoint import load_checkpoint, save_checkpoint, seed_everything


def test_resume_allows_config_changes_but_rejects_different_model(tmp_path):
    import pytest
    from explore_a.config import bind_identity
    task = {"flow": "farneback", "occlusion": "rule", "fusion": "unet_guided"}
    old = {"experiment": task, "input": {"schema": 6, "cache_read": {"probability": .15}},
           "training": {"steps": 60000, "learning_rate": .0001}}
    new = {"experiment": task, "input": {"schema": 2, "manifest": "moved.json"},
           "training": {"steps": 10000, "learning_rate": .0002}}
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, scheduler, scaler, 7, .2, old)
    bind_identity(tmp_path / "identity.json", old)
    bind_identity(tmp_path / "identity.json", new)
    assert (tmp_path / "identity_history.jsonl").exists()
    state = load_checkpoint(path, model, optimizer, scheduler, scaler, new)
    assert state["step"] == 7
    wrong = {**new, "experiment": {**task, "fusion": "final_flow"}}
    with pytest.raises(ValueError, match="model/task"):
        load_checkpoint(path, model, identity=wrong)


def test_online_backend_ignores_optional_remote_config():
    from explore_a.remote_online.policy import settings
    config = {"execution": {"backend": "online"},
              "remote_online": {"enabled": True, "probability": .2}}
    assert settings(config)["enabled"] is False


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
