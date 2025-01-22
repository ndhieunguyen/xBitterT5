from transformers import AutoTokenizer
from src.explainer import xBitterT5_explainer
from src.modeling_t5 import T5ForSequenceClassification
from chemistry_adapters.amino_acids import AminoAcidAdapter
from tqdm import tqdm
import selfies as sf
import pandas as pd
import os


def convert_sequence_to_smiles(sequence):
    adapter = AminoAcidAdapter()
    return adapter.convert_amino_acid_sequence_to_smiles(sequence)


def conver_smiles_to_selfies(smiles):
    return sf.encoder(smiles)


def main(args):
    test_df = pd.read_csv(args.test_csv)
    test_df["smiles"] = test_df.apply(
        lambda row: convert_sequence_to_smiles(row["sequence"]),
        axis=1,
    )
    test_df["selfies"] = test_df.apply(
        lambda row: conver_smiles_to_selfies(row["smiles"]),
        axis=1,
    )

    test_df["sequence"] = test_df.apply(
        lambda row: "<bop>" + "".join("<p>" + aa for aa in row["sequence"]) + "<eop>",
        axis=1,
    )
    test_df["selfies"] = test_df.apply(
        lambda row: "<bom>" + row["selfies"] + "<eom>", axis=1
    )
    test_df["text"] = test_df["sequence"] + test_df["selfies"]
    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()
    num = test_df["No."].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    model = T5ForSequenceClassification.from_pretrained(args.model_ckpt)
    model.eval()
    model.to("cuda")
    cls_explainer = xBitterT5_explainer(model, tokenizer)
    if not os.path.exists(args.save_folder_path):
        os.makedirs(args.save_folder_path)

    pbar = tqdm(range(len(texts)))
    for i in pbar:
        pbar.set_description(f"Processing {num[i]}")
        word_attributions = cls_explainer(texts[i], internal_batch_size=1)

        attribution_filepath = os.path.join(
            args.save_folder_path, f"{num[i]}_{labels[i]}.csv"
        )
        attribution_df = pd.DataFrame(
            {
                "word": [ele[0] for ele in word_attributions],
                "attribution": [ele[1] for ele in word_attributions],
            }
        )
        attribution_df.to_csv(attribution_filepath, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--test_csv", type=str, required=True, help="Path to the test csv file."
    )
    parser.add_argument(
        "--save_folder_path", type=str, required=True, help="Path to the html file."
    )
    args = parser.parse_args()
    main(args)
