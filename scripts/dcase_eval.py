# %%
import os
import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

modes = ["eval", "dev"]
sections = {"dev": [0, 1, 2], "eval": [3, 4, 5]}
domains = ["source", "target", "unseen-target"]
machines = [
    "bearing",
    "fan",
    "gearbox",
    "valve",
    "slider",
    "ToyCar",
    "ToyTrain",
]
no = "002"
if no == "000":
    agg_list = "exp/bearing/asd_model.000/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/fan/asd_model.000/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/gearbox/asd_model.000/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/valve/asd_model.000/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/slider/asd_model.000/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/ToyCar/asd_model.000/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/ToyTrain/asd_model.000/domain-1_seed0/best_loss/best_loss_embed_agg.csv".split(
        " "
    )
elif no == "002":
    agg_list = "exp/bearing/asd_model.002/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/fan/asd_model.002/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/gearbox/asd_model.002/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/valve/asd_model.002/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/slider/asd_model.002/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/ToyCar/asd_model.002/domain-1_seed0/best_loss/best_loss_embed_agg.csv exp/ToyTrain/asd_model.002/domain-1_seed0/best_loss/best_loss_embed_agg.csv".split(
        " "
    )

dev_hauc_cols = []
dev_columns = []
eval_columns = []
use_eval = True
modes = ["dev", "eval"] if use_eval else ["dev"]
sections = {"dev": [0, 1, 2], "eval": [3, 4, 5]}
domains = ["source", "target", "unseen-target"]
for machine in [
    "bearing",
    "fan",
    "gearbox",
    "valve",
    "slider",
    "ToyCar",
    "ToyTrain",
]:
    for domain in domains:
        dev_columns += [
            # f"dev_{machine}_auc",
            # f"dev_{machine}_pauc",
            # f"dev_{domain}_{machine}_hauc",
            f"dev_{domain}_{machine}_auc",
        ]
        if use_eval:
            eval_columns += [
                # f"eval_{machine}_auc",
                # f"eval_{machine}_pauc",
                # f"eval_{domain}_{machine}_hauc",
                f"eval_{domain}_{machine}_auc",
            ]
    # dev_columns += [f"dev_{machine}_hauc"]
    # dev_hauc_cols += [f"dev_{machine}_hauc"]
    dev_columns += [f"dev_{machine}_auc"]
    dev_hauc_cols += [f"dev_{machine}_auc"]
    if use_eval:
        # eval_columns += [f"eval_{machine}_hauc"]
        eval_columns += [f"eval_{machine}_auc"]
# dev_domain_columns = [[], []]
dev_domain_columns = [[], [], []]

for dev_column in dev_columns:
    if "source" in dev_column:
        dev_domain_columns[0].append((dev_column))
    elif "unseen-target" in dev_column:
        dev_domain_columns[2].append((dev_column))
    elif "target" in dev_column:
        dev_domain_columns[1].append((dev_column))
# eval_domain_columns = [[], []]
eval_domain_columns = [[], [], []]
for eval_column in eval_columns:
    if "source" in eval_column:
        eval_domain_columns[0].append((eval_column))
    elif "unseen-target" in eval_column:
        eval_domain_columns[2].append((eval_column))
    elif "target" in eval_column:
        eval_domain_columns[1].append((eval_column))
agg_path = agg_list[0]
agg_df = pd.read_csv(agg_path)
post_processes = list(agg_df.columns)
for rm in ["path", "section", "mode", "is_normal", "domain", "phase"]:
    if rm in post_processes:
        post_processes.remove(rm)
# columns = ["path", "dev_source_hauc", "dev_target_hauc", "dev_hauc"]
columns = [
    "path",
    "dev_source_auc",
    "dev_target_auc",
    "dev_unseen-target_auc",
    "dev_auc",
]
if use_eval:
    # columns += [
    #     "eval_source_hauc",
    #     "eval_target_hauc",
    #     "eval_hauc",
    #     "source_hauc",
    #     "target_hauc",
    #     "hauc",
    # ]
    columns += [
        "eval_source_auc",
        "eval_target_auc",
        "eval_unseen-target_auc",
        "eval_auc",
        "source_auc",
        "target_auc",
        "unseen-target_auc",
        "auc",
    ]
columns += eval_columns + dev_columns
score_df = pd.DataFrame(index=post_processes, columns=columns)
# %%
save_path = os.path.join(
    "/".join(["exp", "all"] + os.path.dirname(agg_path).split("/")[2:]),
    "score_embed.csv",
)
score_df.loc[:, "path"] = save_path
for machine_idx in range(7):
    machine = machines[machine_idx]
    fixed_df = pd.read_csv(
        f"/fsws1/i_kuroyanagi/dcase2023_task2_challenge_recipe_for_github/scripts/downloads/dev/{machine}/blind_attributes_seed0.csv"
    )
    fixed_df = fixed_df[fixed_df["path"].map(lambda x: "test" in x)].reset_index(
        drop=True
    )
    fixed_df["fid"] = fixed_df["fname"].map(lambda x: x.split("_")[5])
    fixed_df.loc[fixed_df["section"] <= 2, "sec_domain_fid"] = (
        fixed_df.loc[fixed_df["section"] <= 2, "section"].astype(str)
        + "_"
        + fixed_df.loc[fixed_df["section"] <= 2, "fname"].map(lambda x: x.split("_")[2])
        + "_"
        + fixed_df.loc[fixed_df["section"] <= 2, "fname"].map(lambda x: x.split("_")[4])
        + "_"
        + fixed_df.loc[fixed_df["section"] <= 2, "fid"].astype(str)
    )
    fixed_df.loc[fixed_df["section"] >= 3, "sec_domain_fid"] = (
        fixed_df.loc[fixed_df["section"] >= 3, "section"].astype(str)
        + "_"
        + fixed_df["path"].map(lambda x: x.split("_")[2])
        + "_"
        + fixed_df["path"].map(lambda x: x.split("_")[4])
        + "_"
        + fixed_df.loc[fixed_df["section"] >= 3, "fid"].astype(str)
    )
    fixed_df_cols = ["sec_domain_fid", "phase", "domain", "state"]
    agg_path = agg_list[machine_idx]
    agg_df = pd.read_csv(agg_path)
    agg_df["fid"] = agg_df["path"].map(lambda x: x.split("_")[5])
    agg_df["sec_domain_fid"] = (
        agg_df["section"].astype(int).astype(str)
        + "_"
        + agg_df["path"].map(lambda x: x.split("_")[2])
        + "_"
        + agg_df["path"].map(lambda x: x.split("_")[4])
        + "_"
        + agg_df["fid"].astype(str)
    )
    # agg_df = pd.merge(agg_df, fixed_df[fixed_df_cols], on="sec_domain_fid", how="left")
    agg_df["is_normal"] = agg_df["path"].map(lambda x: int("normal" in x))
    for post_process in post_processes:
        for mode in modes:
            for domain in domains:
                auc_list = []
                pauc_list = []
                for section in sections[mode]:
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
                    target_idx = (
                        # (agg_df["phase"] == "test")
                        (agg_df["domain"] == domain)
                        & (agg_df["section"] == section)
                    )
                    auc_list.append(
                        roc_auc_score(
                            1 - agg_df.loc[target_idx, "is_normal"],
                            -agg_df.loc[target_idx, post_process],
                        )
                    )
                    # score_df.loc[post_process, f"{mode}_{machine}_auc"] = auc
                    pauc_list.append(
                        roc_auc_score(
                            1 - agg_df.loc[target_idx, "is_normal"],
                            -agg_df.loc[target_idx, post_process],
                            max_fpr=0.1,
                        )
                    )
                    # score_df.loc[post_process, f"{mode}_{machine}_pauc"] = pauc
                score_list = auc_list + pauc_list
                score_df.loc[post_process, f"{mode}_{domain}_{machine}_hauc"] = hmean(
                    score_list
                )
                score_df.loc[post_process, f"{mode}_{domain}_{machine}_auc"] = hmean(
                    auc_list
                )
            auc_list = []
            pauc_list = []
            for section in sections[mode]:
                # target_idx = (agg_df["phase"] == "test") & (
                #     agg_df["section"] == section
                # )
                target_idx = agg_df["section"] == section

                auc_list.append(
                    roc_auc_score(
                        1 - agg_df.loc[target_idx, "is_normal"],
                        -agg_df.loc[target_idx, post_process],
                    )
                )
                # score_df.loc[post_process, f"{mode}_{machine}_auc"] = auc
                pauc_list.append(
                    roc_auc_score(
                        1 - agg_df.loc[target_idx, "is_normal"],
                        -agg_df.loc[target_idx, post_process],
                        max_fpr=0.1,
                    )
                )
                # score_df.loc[post_process, f"{mode}_{machine}_pauc"] = pauc
            score_list = auc_list + pauc_list
            score_df.loc[post_process, f"{mode}_{machine}_hauc"] = hmean(score_list)
            score_df.loc[post_process, f"{mode}_{machine}_auc"] = hmean(auc_list)
for post_process in post_processes:
    for i, domain in enumerate(domains):
        score_df.loc[post_process, f"dev_{domain}_auc"] = hmean(
            score_df.loc[post_process, dev_domain_columns[i]].values.flatten()
        )
        if use_eval:
            score_df.loc[post_process, f"eval_{domain}_auc"] = hmean(
                score_df.loc[post_process, eval_domain_columns[i]].values.flatten()
            )
    score_df.loc[post_process, "dev_auc"] = hmean(
        score_df.loc[post_process, dev_hauc_cols].values.flatten()
    )
    if use_eval:
        score_df.loc[post_process, "eval_auc"] = hmean(
            score_df.loc[post_process, eval_columns].values.flatten()
        )
        score_df.loc[post_process, "auc"] = hmean(
            score_df.loc[post_process, dev_columns + eval_columns].values.flatten()
        )
    # score_df = score_df.reset_index().rename(columns={"index": "post_process"})
    # score_df.to_csv(save_path, index=False)
    # logging.info(f"Successfully saved at {save_path}")

# %%

score_df.to_csv(f"tmp{no}.csv")
# %%
# score_df = pd.read_csv(f"tmp{no}.csv", index_col=0)
domain_idx = 0
domain_auc = pd.DataFrame(
    [
        score_df.loc[
            "GMM_2_all_upper_mean",
            [f"dev_{domains[domain_idx]}_auc"] + dev_domain_columns[domain_idx],
        ].values.flatten(),
        score_df.loc[
            "GMM_2_all_upper_mean",
            [f"eval_{domains[domain_idx]}_auc"] + eval_domain_columns[domain_idx],
        ].values.flatten(),
    ],
    columns=[f"dev_{domains[domain_idx]}_auc"] + dev_domain_columns[domain_idx],
    index=["dev", "eval"],
)
domain_auc
# %%
