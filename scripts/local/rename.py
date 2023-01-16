#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Ibuki Kuroyanagi

"""Rename evaluation dataset."""

import pandas as pd
import os
import shutil


def main():
    machines = ["bearing", "fan", "gearbox", "valve", "slider", "ToyCar", "ToyTrain"]

    for machine in machines:
        for id_ in ["03", "04", "05"]:
            eval_att_path = f"downloads/dcase2022_evaluator/ground_truth_attributes/{machine}/attributes_{id_}.csv"
            gt_df = pd.read_csv(eval_att_path)
            col = "file_name_with_full_attributes"
            if (machine == "valve") and (id_ == "05"):
                gt_df.loc[gt_df[col].isna(), col] = gt_df.loc[gt_df[col].isna(), "d2p"]
            if "Toy" in machine:
                col = "file_name_with_attributes"
            for fname, file_name_with_full_attributes in gt_df.loc[
                :, ["file_name", col]
            ].values:
                original_path = f"downloads/eval/{fname}"
                renamed_path = f"downloads/eval/{file_name_with_full_attributes}"
                if os.path.exists(original_path):
                    os.rename(original_path, renamed_path)
                    print(f"Rename {original_path} to {renamed_path}.")
                if os.path.exists(renamed_path):
                    copy_path = f"downloads/dev/{file_name_with_full_attributes}"
                    shutil.copyfile(renamed_path, copy_path)
                    print(f"Copy {renamed_path} to {copy_path}.")


if __name__ == "__main__":
    main()
