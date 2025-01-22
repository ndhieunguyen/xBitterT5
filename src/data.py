from datasets import Dataset, DatasetDict
import pandas as pd
import glob
import os
import numpy as np


def create_dataset_from_dataframe(
    dataframe_path, pretrained_name, chosen_features=None
):
    dataframe = pd.read_csv(dataframe_path, usecols=["label"] + chosen_features)
    rows_with_nan = dataframe[chosen_features].isna().any(axis=1)
    dataframe = dataframe[np.logical_not(rows_with_nan)]
    if len(chosen_features) > 1:
        for feature in chosen_features:
            if feature == "selfies":
                dataframe[feature] = dataframe.apply(
                    lambda row: "<bom>" + row[feature] + "<eom>", axis=1
                )
            elif feature == "sequence":
                dataframe[feature] = dataframe.apply(
                    lambda row: "<bop>"
                    + "".join("<p>" + aa for aa in row[feature])
                    + "<eop>",
                    axis=1,
                )

            dataframe["text"] = dataframe.apply(
                lambda row: "".join([f"{row[feature]}" for feature in chosen_features]),
                axis=1,
            )

    elif len(chosen_features) == 1:
        chosen_feature = chosen_features[0]
        if chosen_feature == "selfies":
            dataframe["text"] = dataframe.apply(
                lambda row: "<bom>" + row[chosen_feature] + "<eom>", axis=1
            )
        elif chosen_feature == "smiles":
            dataframe["text"] = dataframe[chosen_feature]
        elif chosen_feature == "sequence":
            if "biot5" in pretrained_name:
                dataframe["text"] = dataframe.apply(
                    lambda row: "<bop>"
                    + "".join("<p>" + aa for aa in row[chosen_feature])
                    + "<eop>",
                    axis=1,
                )
            else:
                dataframe["text"] = dataframe.apply(
                    lambda row: " ".join(row[chosen_feature]),
                    axis=1,
                )
    dataframe.drop(columns=chosen_features, inplace=True)
    dataset = Dataset.from_pandas(dataframe)
    return dataset


def create_and_save_datadict(train, val, test, save_path):
    if val is None:
        dataset_dict = DatasetDict({"train": train, "test": test})
        dataset_dict.save_to_disk(save_path)
        return dataset_dict
    dataset_dict = DatasetDict({"train": train, "val": val, "test": test})
    dataset_dict.save_to_disk(save_path)
    return dataset_dict


def prepare_dataset(args):
    fold_folders = glob.glob(args.data_folder + "/fold_*/")
    for fold_folder in fold_folders:
        train_path = os.path.join(fold_folder, "train.csv")
        val_path = os.path.join(fold_folder, "val.csv")
        test_path = os.path.join(fold_folder, "test.csv")

        train = create_dataset_from_dataframe(
            train_path, args.pretrained_name, args.chosen_features
        )
        val = create_dataset_from_dataframe(
            val_path, args.pretrained_name, args.chosen_features
        )
        test = create_dataset_from_dataframe(
            test_path, args.pretrained_name, args.chosen_features
        )
        folder_name = f"dataset_{'_'.join(args.chosen_features)}_{args.pretrained_name.split('/')[-1].replace('-', '_')}"
        save_path = os.path.join(fold_folder, folder_name)
        create_and_save_datadict(train, val, test, save_path)

    train_path = os.path.join(args.data_folder, "train.csv")
    test_path = os.path.join(args.data_folder, "test.csv")
    train = create_dataset_from_dataframe(
        train_path, args.pretrained_name, args.chosen_features
    )
    test = create_dataset_from_dataframe(
        test_path, args.pretrained_name, args.chosen_features
    )
    save_path = os.path.join(
        args.data_folder,
        f"dataset_{'_'.join(args.chosen_features)}_{args.pretrained_name.split('/')[-1].replace('-', '_')}",
    )
    create_and_save_datadict(train, None, test, save_path)
