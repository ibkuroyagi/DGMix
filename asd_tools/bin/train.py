#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Ibuki Kuroyanagi

"""Train Anomalous Sound Detection model."""

import argparse
import logging
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import asd_tools.losses
import asd_tools.models
import asd_tools.optimizers
import asd_tools.schedulers
from asd_tools.datasets import WaveDataset, WaveCollator, WaveEvalCollator
from asd_tools.trainer import DGMixTrainer
from asd_tools.utils import count_params, seed_everything

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train outlier exposure model (See detail in asd_tools/bin/train.py)."
    )
    parser.add_argument(
        "--attribute_csv",
        default=None,
        type=str,
        help="Path of attribute csv.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.getLogger("matplotlib.font_manager").disabled = True
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    seed_everything(seed=config["seed"])
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    df = pd.read_csv(args.attribute_csv)
    use_cols = ["path", "domain", "phase", "state", "machine", "section"]
    # for train idx
    train_dataset = WaveDataset(
        df.loc[df["phase"] == "train", use_cols],
        allow_cache=config.get("allow_cache", False),
    )
    # for valid idx
    valid_dataset = WaveDataset(
        df.loc[df["phase"] == "valid", use_cols],
        allow_cache=True,
    )
    # for visualization idx
    n_plot = 120
    pos_test_idx = np.where((df["phase"] == "test") & (df["section"] >= 3))[0]
    pos_valid_idx = np.where(
        (df["phase"] == "valid") & (df["domain"] == "source") & (df["section"] >= 3)
    )[0]
    pos_valid_idx = np.random.choice(pos_valid_idx, min(n_plot, len(pos_valid_idx)))
    pos_train_idx = np.where(
        (df["phase"] == "train") & (df["domain"] == "source") & (df["section"] >= 3)
    )[0]
    pos_train_idx = np.random.choice(pos_train_idx, min(n_plot, len(pos_train_idx)))
    pos_target_idx = np.where(
        df["phase"].map(lambda x: x in ["train", "valid"])
        & (df["domain"] == "target")
        & (df["section"] >= 3)
    )[0]
    pos_idx = np.concatenate(
        [pos_test_idx, pos_valid_idx, pos_train_idx, pos_target_idx],
        0,
    )
    visualize_dataset = WaveDataset(
        df.loc[pos_idx, use_cols],
        allow_cache=True,
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"The number of validation files = {len(valid_dataset)}.")
    logging.info(f"The number of visualization files = {len(visualize_dataset)}.")
    train_collator = WaveCollator(
        sf=config["sf"],
        sec=config["sec"],
        shuffle=True,
    )
    valid_collator = WaveCollator(
        sf=config["sf"],
        sec=config["sec"],
        shuffle=False,
    )
    visualize_collator = WaveEvalCollator(
        sf=config["sf"],
        sec=config["sec"],
    )
    data_loader = {
        "train": DataLoader(
            dataset=train_dataset,
            collate_fn=train_collator,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=True,
            drop_last=False,
        ),
        "valid": DataLoader(
            valid_dataset,
            collate_fn=valid_collator,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
            drop_last=False,
        ),
        "visualize": DataLoader(
            visualize_dataset,
            collate_fn=visualize_collator,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
            drop_last=False,
        ),
    }

    # define models and optimizers
    model_class = getattr(asd_tools.models, config["model_type"])
    model = model_class(**config["model_params"]).to(device)
    logging.info(model)
    params_cnt = count_params(model)
    logging.info(f"Size of model is {params_cnt}.")
    optimizer_class = getattr(
        asd_tools.optimizers,
        config["optimizer_type"],
    )
    params_list = [{"params": model.parameters()}]
    optimizer = optimizer_class(params_list, **config["optimizer_params"])
    if config.get("scheduler_type", None) is not None:
        scheduler_class = getattr(
            asd_tools.schedulers,
            config["scheduler_type"],
        )
        scheduler = scheduler_class(optimizer=optimizer, **config["scheduler_params"])

    # define trainer
    trainer = DGMixTrainer(
        steps=1,
        epochs=1,
        data_loader=data_loader,
        model=model.to(device),
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        train=True,
        stage="pretrain",
    )
    # run training loop
    trainer.run()

    # define finetune settings
    data_loader = {
        "train": DataLoader(
            dataset=train_dataset,
            collate_fn=train_collator,
            batch_size=config["finetune_batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=True,
            drop_last=False,
        ),
        "valid": DataLoader(
            valid_dataset,
            collate_fn=valid_collator,
            batch_size=config["finetune_batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
            drop_last=False,
        ),
        "visualize": DataLoader(
            visualize_dataset,
            collate_fn=visualize_collator,
            batch_size=config["finetune_batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
            drop_last=False,
        ),
    }
    optimizer_class = getattr(
        asd_tools.optimizers,
        config["finetune_optimizer_type"],
    )
    params_list = [{"params": trainer.model.parameters()}]
    optimizer = optimizer_class(params_list, **config["finetune_optimizer_params"])
    config["train_max_epochs"] += config["finetune_epochs"]
    config["save_interval_epochs"] = config["finetune_epochs"]
    # define fine tune trainer
    trainer = DGMixTrainer(
        steps=trainer.steps,
        epochs=trainer.epochs,
        data_loader=data_loader,
        model=trainer.model.to(device),
        optimizer=optimizer,
        scheduler=None,
        config=config,
        device=device,
        train=True,
        stage="finetune",
    )
    # run fine tune training loop
    trainer.run()


if __name__ == "__main__":
    main()
