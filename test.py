from src.modeling_t5 import T5ForSequenceClassification
import selfies as sf
import pandas as pd
from transformers import AutoTokenizer, pipeline
from chemistry_adapters.amino_acids import AminoAcidAdapter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef as mcc
from tqdm import tqdm


def convert_sequence_to_smiles(sequence):
    adapter = AminoAcidAdapter()
    return adapter.convert_amino_acid_sequence_to_smiles(sequence)


def conver_smiles_to_selfies(smiles):
    return sf.encoder(smiles)


def main(args):

    test_df = pd.read_csv(args.test_csv)
    y_true = test_df["label"].values

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

    text_inputs = test_df["text"].tolist()
    y_true = test_df["label"].tolist()

    model = T5ForSequenceClassification.from_pretrained(args.model_ckpt)
    model.eval()
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)

    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, device=args.device
    )

    batch_size = args.batch_size
    result = []
    for i in tqdm(range(0, len(text_inputs), batch_size)):
        batch = text_inputs[i : i + batch_size]
        result.extend(classifier(batch))

    y_pred, y_prob = [], []
    for pred in result:
        if pred["label"] == "bitter":
            y_prob.append(pred["score"])
            y_pred.append(1)
        else:
            y_prob.append(1 - pred["score"])
            y_pred.append(0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    fpr2, tpr2, _ = roc_curve(y_true, y_prob, pos_label=1)
    auc2 = auc(fpr2, tpr2)
    mcc_score = mcc(y_true, y_pred)

    print(f"Classification report: {classification_report(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"F1 score: {f1_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred)}")
    print(f"Recall: {recall_score(y_true, y_pred)}")
    print(f"Confusion matrix: {confusion_matrix(y_true, y_pred)}")
    print(f"Specificity: {specificity}")
    print(f"AUC: {auc2}")
    print(f"MCC: {mcc_score}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)
