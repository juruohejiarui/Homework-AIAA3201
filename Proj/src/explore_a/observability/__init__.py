"""Observational UI and derived artifacts; never used for optimization or model selection."""

DEFAULTS = {"progress": "auto", "tensorboard": True, "plots": True, "plot_every": 1000,
            "refresh_seconds": 1.0, "non_tty_every_s": 30.0, "tb_flush_s": 10}


def settings(config):
    return {**DEFAULTS, **config.get("observability", {})}
