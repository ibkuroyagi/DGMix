# %%
# %load_ext autoreload
# %autoreload 2
import torch
import pandas as pd
from scipy.stats import hmean
from IPython.display import display

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
# %%
# disentangle method
score_df = pd.read_csv(
    "exp/all/disentangle.original/domain-1_seed0/checkpoint-250epochs/score.csv"
)
domain = domains[2]
for domain in domains:
    cols = [f"AUC_{domain}"]
    dev_cols = [f"dev_AUC_{domain}"]
    eval_cols = [f"eval_AUC_{domain}"]
    post_process = "KNN_1_all"
    for machine in machines:
        dev_cols.append(f"dev_{domain}_{machine}_auc")
        eval_cols.append(f"eval_{domain}_{machine}_auc")
        cols.append(f"{domain}_{machine}_auc")
    dev_df = score_df.loc[score_df["post_process"] == post_process, dev_cols]
    eval_df = score_df.loc[score_df["post_process"] == post_process, eval_cols]
    compare_df = pd.DataFrame(
        [dev_df.values.flatten(), eval_df.values.flatten()],
        columns=cols,
        index=["dev", "eval"],
    )
    display(compare_df * 100)
# %%
# dgmix method
score_df = pd.read_csv(
    "exp/all/dgmix.original/domain-1_seed0/checkpoint-150epochs/score.csv"
)
domain = domains[0]
for domain in domains:
    cols = [f"AUC_{domain}"]
    dev_cols = [f"dev_AUC_{domain}"]
    eval_cols = [f"eval_AUC_{domain}"]
    post_process = "KNN_1_all"
    for machine in machines:
        dev_cols.append(f"dev_{domain}_{machine}_auc")
        eval_cols.append(f"eval_{domain}_{machine}_auc")
        cols.append(f"{domain}_{machine}_auc")
    dev_df = score_df.loc[score_df["post_process"] == post_process, dev_cols]
    eval_df = score_df.loc[score_df["post_process"] == post_process, eval_cols]
    compare_df = pd.DataFrame(
        [dev_df.values.flatten(), eval_df.values.flatten()],
        columns=cols,
        index=["dev", "eval"],
    )
    display(compare_df * 100)
# %%
# proposed method
score_df = pd.read_csv(
    "exp/all/asd_model.000/domain-1_seed0/checkpoint-250epochs/score_embed.csv"
)
domain = domains[1]
cols = [f"AUC_{domain}"]
dev_cols = [f"dev_AUC_{domain}"]
eval_cols = [f"eval_AUC_{domain}"]
post_process = "KNN_2_all_upper_mean"
for machine in machines:
    dev_cols.append(f"dev_{domain}_{machine}_auc")
    eval_cols.append(f"eval_{domain}_{machine}_auc")
    cols.append(f"{domain}_{machine}_auc")
dev_df = score_df.loc[score_df["post_process"] == post_process, dev_cols]
eval_df = score_df.loc[score_df["post_process"] == post_process, eval_cols]
compare_df = pd.DataFrame(
    [dev_df.values.flatten(), eval_df.values.flatten()],
    columns=cols,
    index=["dev", "eval"],
)
compare_df

# %%
score_df = pd.read_csv(
    "exp/all/asd_model.000/domain-1_seed0/checkpoint-300epochs/score_embed.csv"
)
score_df["agg"] = score_df["post_process"].map(lambda x: x.split("_")[3])
domain = domains[2]
cols = [f"AUC_{domain}"]
dev_all_list = []
eval_all_list = []
for machine in machines:
    cols.append(f"{domain}_{machine}_auc")
    dev_auc_pauc_list = []
    eval_auc_pauc_list = []
    for i in range(3):
        tmp = score_df[score_df["agg"] == "upper"].sort_values(
            by=f"{machine}_id{sections['dev'][i]}_auc", ascending=False
        )
        tmp.reset_index(drop=True, inplace=True)
        dev_auc_pauc_list += list(
            tmp.loc[
                0,
                [
                    f"{domain}_{machine}_id{sections['dev'][i]}_auc",
                    f"{domain}_{machine}_id{sections['dev'][i]}_pauc",
                ],
            ].values
        )
        eval_auc_pauc_list += list(
            tmp.loc[
                0,
                [
                    f"{domain}_{machine}_id{sections['eval'][i]}_auc",
                    f"{domain}_{machine}_id{sections['eval'][i]}_pauc",
                ],
            ].values
        )
    dev_all_list.append(hmean(dev_auc_pauc_list))
    eval_all_list.append(hmean(eval_auc_pauc_list))
dev_all = hmean(dev_all_list)
dev_all_list.insert(0, dev_all)
eval_all = hmean(eval_all_list)
eval_all_list.insert(0, eval_all)
compare_df = pd.DataFrame(
    [dev_all_list, eval_all_list], columns=cols, index=["dev", "eval"]
)
compare_df
# %%
df = pd.read_csv(
    "/fsws1/i_kuroyanagi/DGMix/scripts/downloads/dev/bearing/attributes_seed0.csv"
)
df[["phase", "section", "domain", "state"]].value_counts()
# %%
