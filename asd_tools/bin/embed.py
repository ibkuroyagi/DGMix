#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Ibuki Kuroyanagi

"""Extract embedding vectors."""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import asd_tools.models
from asd_tools.datasets import WaveEvalCollator, WaveDataset


def main():
    """Extract embedding features."""
    parser = argparse.ArgumentParser(description="Extract embedding features.")
    parser.add_argument(
        "--attribute_csv",
        default=None,
        type=str,
        help="Path of attribute csv.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--checkpoints",
        default=[],
        type=str,
        nargs="+",
        help="checkpoint file path to resume training. (default=[])",
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

    for key, value in config.items():
        logging.info(f"{key} = {value}")
    attribute_df = pd.read_csv(args.attribute_csv)
    # get dataset
    infer_dataset = WaveDataset(
        attribute_df,
        allow_cache=False,
    )
    logging.info(f"The number of infer files = {len(infer_dataset)}.")
    collator = WaveEvalCollator(
        sf=config["sf"],
        sec=config["sec"],
    )
    loader = DataLoader(
        infer_dataset,
        batch_size=config["batch_size"],
        collate_fn=collator,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        shuffle=False,
        drop_last=False,
    )
    embed_size = config["model_params"]["embedding_size"]
    for checkpoint in args.checkpoints:
        model_class = getattr(asd_tools.models, config["model_type"])
        model = model_class(**config["model_params"])
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        model.to(device)
        model.eval()
        logging.info(f"Successfully loaded {checkpoint}.")
        logging.info(
            f"Steps:{state_dict['steps']}, "
            f"Epochs:{state_dict['epochs']}, "
            f"BEST loss:{state_dict['best_loss']}"
        )
        path_list = np.empty(0)
        embed = np.empty((0, embed_size))

        for batch in loader:
            with torch.no_grad():
                y_ = model(batch["wave"].to(device), get_only_embed=True)
                batch_embed = y_["embedding"]
            path_list = np.append(path_list, batch["path"])
            embed = np.concatenate([embed, batch_embed.cpu().numpy()], axis=0)
        embed_cols = [f"e{i}" for i in range(embed_size)]
        columns = ["path"] + embed_cols
        df = pd.DataFrame(embed, columns=embed_cols)
        df["path"] = path_list
        df = df[columns]
        logging.info(f"Shape of df is {df.shape}.")
        ckpt_df = pd.merge(attribute_df, df, how="left", on="path")
        csv_path = checkpoint.replace(".pkl", ".csv")
        ckpt_df.to_csv(csv_path, index=False)
        logging.info(f"Saved at {csv_path}")


if __name__ == "__main__":
    main()
