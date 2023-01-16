#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Ibuki Kuroyanagi

"""Inference models."""

import argparse
import logging
import os
import sys
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors
from asd_tools.utils import seed_everything


def main():
    """Run inference process."""
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--attribute_csv",
        default=None,
        type=str,
        help="Path of attribute csv.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed",
    )
    parser.add_argument(
        "--checkpoints",
        default=[],
        type=str,
        nargs="+",
        help="Extracted embedding file for validation.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()
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
    n_section = 6
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    embed_size = config["model_params"]["embedding_size"]
    knn_metric = "euclidean"
    feature_cols = [f"e{i}" for i in range(embed_size)]
    checkpoint_dirs = [os.path.dirname(checkpoint) for checkpoint in args.checkpoints]
    for checkpoint_dir in checkpoint_dirs:
        post_cols = []
        checkpoint = checkpoint_dir.split("/")[-1]
        logging.info(f"checkpoint_dir:{checkpoint_dir}")
        att_path = os.path.join(checkpoint_dir, checkpoint + ".csv")
        attribute_df = pd.read_csv(att_path)
        attribute_df["is_anomaly"] = attribute_df["path"].map(
            lambda x: int("anomaly" in x)
        )
        for section_id in range(n_section):
            for used_set in ["all", "valid"]:
                if used_set == "all":
                    train_idx = (
                        (attribute_df["phase"] == "valid")
                        & (attribute_df["section"] == section_id)
                    ) | (
                        (attribute_df["phase"] == "train")
                        & (attribute_df["section"] == section_id)
                    )
                elif used_set == "valid":
                    train_idx = (attribute_df["phase"] == "valid") & (
                        attribute_df["section"] == section_id
                    )
                test_idx = (attribute_df["section"] == section_id) & (
                    attribute_df["phase"] == "test"
                )
                train_df = attribute_df.loc[train_idx, feature_cols]
                test_df = attribute_df.loc[test_idx, feature_cols]
                knn = NearestNeighbors(n_neighbors=1, metric=knn_metric)
                knn.fit(train_df)
                knn_score = knn.kneighbors(test_df)[0].mean(1)
                attribute_df.loc[test_idx, f"KNN_1_{used_set}"] = knn_score
                if section_id == 0:
                    post_cols += [
                        f"KNN_1_{used_set}",
                    ]
                post_cols.sort()
        attribute_df.loc[attribute_df["section"] <= 2, "mode"] = "dev"
        attribute_df.loc[attribute_df["section"] >= 3, "mode"] = "eval"
        columns = [
            "path",
            "section",
            "mode",
            "phase",
            "is_anomaly",
            "domain",
        ] + post_cols
        infer_df = attribute_df[columns]
        logging.info(infer_df.columns)
        infer_path = os.path.join(checkpoint_dir, checkpoint + "_agg.csv")
        infer_df.to_csv(infer_path, index=False)
        logging.info(f"Successfully saved {infer_path}.")


if __name__ == "__main__":
    main()
