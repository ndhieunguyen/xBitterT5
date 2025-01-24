from src.modeling_t5 import T5ForSequenceClassification
import selfies as sf
import pandas as pd
from transformers import AutoTokenizer, pipeline
from chemistry_adapters.amino_acids import AminoAcidAdapter
from tqdm import tqdm


class xBitterT5_predictor:
    def __init__(
        self,
        xBitterT5_640_ckpt="cbbl-skku-org/xBitterT5-640",
        xBitterT5_720_ckpt="cbbl-skku-org/xBitterT5-720",
        device="cuda",
    ):
        self.xBitterT5_640_ckpt = xBitterT5_640_ckpt
        self.xBitterT5_720_ckpt = xBitterT5_720_ckpt
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(xBitterT5_640_ckpt)
        self.xBitterT5_640 = self.load_model(xBitterT5_640_ckpt)
        self.xBitterT5_720 = self.load_model(xBitterT5_720_ckpt)

        self.classifier_640 = pipeline(
            "text-classification",
            model=self.xBitterT5_640,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        self.classifier_720 = pipeline(
            "text-classification",
            model=self.xBitterT5_720,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def load_model(self, ckpt):
        model = T5ForSequenceClassification.from_pretrained(ckpt)
        model.eval()
        model.to(self.device)
        return model

    def convert_sequence_to_smiles(self, sequence):
        adapter = AminoAcidAdapter()
        return adapter.convert_amino_acid_sequence_to_smiles(sequence)

    def conver_smiles_to_selfies(self, smiles):
        return sf.encoder(smiles)

    def predict(
        self,
        input_dict,
        model_type="xBitterT5-720",
        batch_size=4,
    ):
        assert model_type in ["xBitterT5-640", "xBitterT5-720"]
        df = pd.DataFrame(
            {"id": list(input_dict.keys()), "sequence": list(input_dict.values())}
        )

        df["smiles"] = df.apply(
            lambda row: self.convert_sequence_to_smiles(row["sequence"]),
            axis=1,
        )
        df["selfies"] = df.apply(
            lambda row: self.conver_smiles_to_selfies(row["smiles"]),
            axis=1,
        )

        df["sequence"] = df.apply(
            lambda row: "<bop>"
            + "".join("<p>" + aa for aa in row["sequence"])
            + "<eop>",
            axis=1,
        )
        df["selfies"] = df.apply(lambda row: "<bom>" + row["selfies"] + "<eom>", axis=1)
        df["text"] = df["sequence"] + df["selfies"]

        text_inputs = df["text"].tolist()

        if model_type == "xBitterT5-640":
            classifier = self.classifier_640
        else:
            classifier = self.classifier_720

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

        return {i: [y_prob[j], y_pred[j]] for j, i in enumerate(df["id"].tolist())}


if __name__ == "__main__":
    import time

    predictor = xBitterT5_predictor(
        device="cpu",
        xBitterT5_640_ckpt="output/xBitterT5-640",
        xBitterT5_720_ckpt="output/xBitterT5-720",
    )
    input_dict = {1: "PA"}
    start = time.time()
    result = predictor.predict(input_dict)
    print(f"Time take in seconds: {time.time() - start}")
    print(result)
