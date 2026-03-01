"""Experiment configuration for remote deployment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LambdaConfig:
    host: str = ""
    ssh_key: str = "~/.ssh/id_ed25519_goblin"
    ssh_user: str = "ubuntu"
    instance_id: str = ""
    api_key: str = ""

    @property
    def ssh_key_path(self) -> str:
        return os.path.expanduser(self.ssh_key)

    @property
    def connection_string(self) -> str:
        return f"{self.ssh_user}@{self.host}"


@dataclass
class ExperimentConfig:
    animal: str = "dogs"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    data_count: int = 10000
    epochs: int = 10
    seeds: str = "42"
    effective_batch_size: int = 60
    per_device_batch_size: int = 6
    learning_rate: float = 0.0002
    warmup_steps: int = 5
    max_seq_length: int = 256
    max_train_samples: int | None = None
    logging_steps: int = 50
    save_total_limit: int = 2
    checkpoint_upload_every_n_epochs: int = 2
    repo_url: str = "https://github.com/rain-1/subliminal-learning-replication.git"
    remote_dir: str = "~/subliminal-learning-replication"


@dataclass
class HubConfig:
    repo_prefix: str = ""
    push_checkpoints: bool = True
    private: bool = False
    token: str = ""


@dataclass
class WandbConfig:
    project: str = "subliminal-learning"
    tags: str = ""
    group: str = "subliminal-learning-replication"
    entity: str = ""


@dataclass
class Config:
    lambda_: LambdaConfig = field(default_factory=LambdaConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} or $VAR with environment variable values."""
    def _replace(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))
    return re.sub(r"\$\{(\w+)\}|\$(\w+)", _replace, value)


def _resolve_dict(d: dict) -> dict:
    """Recursively resolve env vars in string values."""
    resolved = {}
    for k, v in d.items():
        if isinstance(v, dict):
            resolved[k] = _resolve_dict(v)
        elif isinstance(v, str):
            resolved[k] = _resolve_env_vars(v)
        else:
            resolved[k] = v
    return resolved


def load_config(path: str | Path) -> Config:
    """Load experiment config from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    raw = _resolve_dict(raw or {})

    cfg = Config()

    if "lambda" in raw:
        lc = raw["lambda"]
        cfg.lambda_ = LambdaConfig(
            host=lc.get("host", ""),
            ssh_key=lc.get("ssh_key", cfg.lambda_.ssh_key),
            ssh_user=lc.get("ssh_user", cfg.lambda_.ssh_user),
            instance_id=lc.get("instance_id", ""),
            api_key=lc.get("api_key", ""),
        )

    if "experiment" in raw:
        ec = raw["experiment"]
        cfg.experiment = ExperimentConfig(
            animal=ec.get("animal", cfg.experiment.animal),
            base_model=ec.get("base_model", cfg.experiment.base_model),
            data_count=ec.get("data_count", cfg.experiment.data_count),
            epochs=ec.get("epochs", cfg.experiment.epochs),
            seeds=str(ec.get("seeds", cfg.experiment.seeds)),
            effective_batch_size=ec.get("effective_batch_size", cfg.experiment.effective_batch_size),
            per_device_batch_size=ec.get("per_device_batch_size", cfg.experiment.per_device_batch_size),
            learning_rate=ec.get("learning_rate", cfg.experiment.learning_rate),
            warmup_steps=ec.get("warmup_steps", cfg.experiment.warmup_steps),
            max_seq_length=ec.get("max_seq_length", cfg.experiment.max_seq_length),
            max_train_samples=ec.get("max_train_samples", None),
            logging_steps=ec.get("logging_steps", cfg.experiment.logging_steps),
            save_total_limit=ec.get("save_total_limit", cfg.experiment.save_total_limit),
            checkpoint_upload_every_n_epochs=ec.get("checkpoint_upload_every_n_epochs", cfg.experiment.checkpoint_upload_every_n_epochs),
            repo_url=ec.get("repo_url", cfg.experiment.repo_url),
            remote_dir=ec.get("remote_dir", cfg.experiment.remote_dir),
        )

    if "hub" in raw:
        hc = raw["hub"]
        cfg.hub = HubConfig(
            repo_prefix=hc.get("repo_prefix", ""),
            push_checkpoints=hc.get("push_checkpoints", True),
            private=hc.get("private", False),
            token=hc.get("token", ""),
        )

    if "wandb" in raw:
        wc = raw["wandb"]
        cfg.wandb = WandbConfig(
            project=wc.get("project", cfg.wandb.project),
            tags=wc.get("tags", ""),
            group=wc.get("group", cfg.wandb.group),
            entity=wc.get("entity", ""),
        )

    return cfg
