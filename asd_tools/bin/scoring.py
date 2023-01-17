#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Ibuki Kuroyanagi

"""Scoring."""

import argparse
import logging
import os
import sys
import warnings

import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

warnings.simplefilter("ignore")
domains = ["source", "target"]
machines = [
    "bearing",
    "fan",
    "gearbox",
    "valve",
    "slider",
    "ToyCar",
    "ToyTrain",
]


def get_args():
    parser = argparse.ArgumentParser(description="Scoring script.")
    parser.add_argument(
        "--agg_checkpoints",
        default=[],
        type=str,
        nargs="+",
        help="Aggregated checkpoints files.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument("--concat", action="store_true")
    args = parser.parse_args()
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


def main(args):
    """Run scoring process."""
    modes = ["eval", "dev"]
    sections = {"dev": [0, 1, 2], "eval": [3, 4, 5]}

    # for dcase 2023 metric (AUC_machine_section_domain=7*3*3domain, pAUC_machine_section=7*3)
    dcase2022_cols_dict = {
        "eval_official": [],  # harmonic mean of source, target, and pAUC
        "eval_AUC_source": [],
        "eval_AUC_target": [],
        "eval_pAUC": [],
        "dev_official": [],  # harmonic mean of source, target, and pAUC
        "dev_AUC_source": [],
        "dev_AUC_target": [],
        "dev_pAUC": [],
    }

    for mode in modes:
        for machine in machines:
            for section in sections[mode]:
                dcase2022_cols_dict[f"{mode}_pAUC"].append(
                    f"{machine}_id{section}_pauc"
                )
                for domain in domains:
                    if domain == "unseen-target":
                        if (
                            ((machine == "bearing") and (section in [2, 5]))
                            or ((machine == "fan") and (section in [1, 4]))
                            # or ((machine == "gearbox") and (section in []))
                            or ((machine == "slider") and (section in [2, 5]))
                            or ((machine == "valve") and (section in [2, 5]))
                            or ((machine == "ToyCar") and (section in [1, 2, 4]))
                            or ((machine == "ToyTrain") and (section in [1, 2, 4]))
                        ):
                            continue
                    dcase2022_cols_dict[f"{mode}_AUC_{domain}"].append(
                        f"{domain}_{machine}_id{section}_auc"
                    )
        dcase2022_cols_dict[f"{mode}_official"] += (
            dcase2022_cols_dict[f"{mode}_pAUC"]
            + dcase2022_cols_dict[f"{mode}_AUC_source"]
            + dcase2022_cols_dict[f"{mode}_AUC_target"]
        )

    for k, v in dcase2022_cols_dict.items():
        logging.info(f"dcase2022_cols_dict: {k} {len(v)}")

    agg_df = pd.read_csv(args.agg_checkpoints[0])
    post_processes = list(agg_df.columns)
    for rm in ["path", "is_anomaly", "section", "mode", "domain", "phase"]:
        post_processes.remove(rm)
    score_df = pd.DataFrame(index=post_processes, columns=["path"])
    save_path = os.path.join(
        "/".join(
            ["exp", "all"] + os.path.dirname(args.agg_checkpoints[0]).split("/")[2:]
        ),
        "score.csv",
    )
    save_md_path = os.path.join(
        "/".join(
            ["exp", "all"] + os.path.dirname(args.agg_checkpoints[0]).split("/")[2:]
        ),
        "score.md",
    )
    score_df.loc[:, "path"] = save_path
    for agg_path in args.agg_checkpoints:
        logging.info(f"Loaded {agg_path}.")
        agg_df = pd.read_csv(agg_path)
        machine = agg_path.split("/")[1]
        for post_process in post_processes:
            for mode in modes:
                for domain in domains:
                    auc_list = []
                    pauc_list = []
                    mauc = 1.1
                    for section in sections[mode]:
                        target_idx = (
                            (agg_df["domain"] == domain)
                            & (agg_df["section"] == section)
                            & (agg_df["phase"] == "test")
                        )

                        auc = roc_auc_score(
                            agg_df.loc[target_idx, "is_anomaly"],
                            agg_df.loc[target_idx, post_process],
                        )
                        auc_list.append(auc)
                        score_df.loc[
                            post_process, f"{domain}_{machine}_id{section}_auc"
                        ] = auc
                        if mauc > auc:
                            mauc = auc
                        pauc = roc_auc_score(
                            agg_df.loc[target_idx, "is_anomaly"],
                            agg_df.loc[target_idx, post_process],
                            max_fpr=0.1,
                        )
                        pauc_list.append(pauc)
                        score_df.loc[
                            post_process, f"{domain}_{machine}_id{section}_pauc"
                        ] = pauc
                    score_df.loc[post_process, f"{mode}_{domain}_{machine}_mauc"] = mauc
                    score_list = auc_list + pauc_list
                    score_df.loc[
                        post_process, f"{mode}_{domain}_{machine}_hauc"
                    ] = hmean(score_list)
                    score_df.loc[
                        post_process, f"{mode}_{domain}_{machine}_auc"
                    ] = hmean(auc_list)
                auc_list = []
                pauc_list = []
                mauc = 1.1
                for section in sections[mode]:
                    target_idx = (agg_df["section"] == section) & (
                        agg_df["phase"] == "test"
                    )

                    auc = roc_auc_score(
                        agg_df.loc[target_idx, "is_anomaly"],
                        agg_df.loc[target_idx, post_process],
                    )
                    auc_list.append(auc)
                    score_df.loc[post_process, f"{machine}_id{section}_auc"] = auc
                    if mauc > auc:
                        mauc = auc
                    pauc = roc_auc_score(
                        agg_df.loc[target_idx, "is_anomaly"],
                        agg_df.loc[target_idx, post_process],
                        max_fpr=0.1,
                    )
                    pauc_list.append(pauc)
                    score_df.loc[post_process, f"{machine}_id{section}_pauc"] = pauc
                score_df.loc[post_process, f"{mode}_{machine}_mauc"] = mauc
                score_list = auc_list + pauc_list
                score_df.loc[post_process, f"{mode}_{machine}_hauc"] = hmean(score_list)
    for post_process in post_processes:
        for col_name, col_list in dcase2022_cols_dict.items():
            score_df.loc[post_process, col_name] = hmean(
                score_df.loc[post_process, col_list].values.flatten()
            )

    score_df = score_df.reset_index().rename(columns={"index": "post_process"})
    score_df.to_csv(save_path, index=False)
    logging.info(f"Successfully saved at {save_path}")
    write_md(score_df, save_md_path)


def concat_scores(args):
    df_list = []
    for agg_checkpoint in args.agg_checkpoints:
        logging.info(f"Loaded file is {agg_checkpoint}.")
        df_list.append(pd.read_csv(agg_checkpoint))
    score_df = pd.concat(df_list, axis=0)
    save_path = "/".join(agg_checkpoint.split("/")[:-2] + ["score.csv"])
    score_df.to_csv(save_path, index=False)
    logging.info(f"Concatenated file is saved at {save_path}.")
    save_md_path = "/".join(agg_checkpoint.split("/")[:-2] + ["score.md"])
    write_md(score_df, save_md_path)


def write_md(score_df, save_md_path):
    cols = ["AUC"]
    df_list = []
    for domain in domains:
        dev_cols = [f"dev_AUC_{domain}"]
        eval_cols = [f"eval_AUC_{domain}"]
        post_process = "KNN_1_all"
        for machine in machines:
            dev_cols.append(f"dev_{domain}_{machine}_auc")
            eval_cols.append(f"eval_{domain}_{machine}_auc")
            if domain == "source":
                cols.append(f"{machine}_auc")
        dev_df = score_df.loc[score_df["post_process"] == post_process, dev_cols]
        eval_df = score_df.loc[score_df["post_process"] == post_process, eval_cols]
        df_list.append(
            pd.DataFrame(
                [dev_df.values.flatten(), eval_df.values.flatten()],
                columns=cols,
                index=[f"{domain}_dev", f"{domain}_eval"],
            )
            * 100
        )

    compare_df = pd.concat(df_list, axis=0)
    md_txt = compare_df.to_markdown()
    logging.info(md_txt)
    with open(save_md_path, mode="w") as f:
        f.write(md_txt)
    logging.info(f"Successfully saved at {save_md_path}")


if __name__ == "__main__":
    args = get_args()
    if args.concat:
        concat_scores(args)
    else:
        main(args)
