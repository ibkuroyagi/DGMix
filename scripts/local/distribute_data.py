#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Ibuki Kuroyanagi

"""Distribute data to make unseen domain data in evaluation data."""

import argparse
import logging
import sys
import warnings
from glob import glob

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from asd_tools.utils import seed_everything

warnings.simplefilter("ignore")


def get_args():
    """Distribute data."""
    parser = argparse.ArgumentParser(
        description="Distribute data to make unseen domain data in evaluation data."
    )
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--n_min_low_target",
        type=int,
        default=3,
        help="The number of low target domain data which are contained to training data.",
    )
    parser.add_argument(
        "--valid_percent",
        type=int,
        default=15,
        help="The percent of validation data.",
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
    return args


def main():
    n_section = 6
    args = get_args()
    seed_everything(args.seed)
    machines = ["bearing", "fan", "gearbox", "slider", "valve", "ToyCar", "ToyTrain"]
    #  concatenate all IDs attribute csv
    for machine in machines:
        df_list = []
        for id_ in range(n_section):
            path_df = pd.DataFrame(
                sorted(list(glob(f"downloads/dev/{machine}/**/section_0{id_}*.wav"))),
                columns=["path"],
            )
            path_df["file_name"] = path_df["path"].map(
                lambda x: x.replace("downloads/dev/", "")
            )
            path_df["phase"] = path_df["path"].map(lambda x: x.split("/")[3])

            df_list.append(path_df)
        df = pd.concat(df_list)
        df.reset_index(inplace=True, drop=True)
        df["machine"] = df["file_name"].map(lambda x: x.split("/")[0])
        df["section"] = df["file_name"].map(lambda x: int(x.split("_")[1]))
        df["phase"] = df["file_name"].map(lambda x: x.split("/")[1])
        df["domain"] = df["file_name"].map(lambda x: x.split("_")[2])
        df["state"] = df["file_name"].map(lambda x: x.split("_")[4])
        skf2 = StratifiedKFold(
            n_splits=int(100 / args.valid_percent),
            shuffle=True,
            random_state=args.seed,
        )
        section = df.loc[df["phase"] == "train", "section"]
        for _, valid_index in skf2.split(section, section):
            idx = df.loc[df["phase"] == "train", :].index[valid_index]
            df.loc[idx, "phase"] = "valid"
            break
        df.to_csv(
            f"downloads/dev/{machine}/attributes_seed{args.seed}.csv", index=False
        )


if __name__ == "__main__":
    main()
