# # First phase: load model and tokenizer then save to local
# import torch
# from src.modeling_t5 import T5ForSequenceClassification
# from transformers import AutoTokenizer

# xBitterT5_640_ckpt = "cbbl-skku-org/xBitterT5-640"
# xBitterT5_720_ckpt = "cbbl-skku-org/xBitterT5-720"

# tokenizer = AutoTokenizer.from_pretrained(xBitterT5_640_ckpt)
# xBitterT5_640 = T5ForSequenceClassification.from_pretrained(xBitterT5_640_ckpt)
# xBitterT5_720 = T5ForSequenceClassification.from_pretrained(xBitterT5_720_ckpt)

# tokenizer.save_pretrained("tokenizer")
# torch.save(xBitterT5_640.state_dict(), "model_640.pt")
# torch.save(xBitterT5_720.state_dict(), "model_720.pt")

#########################################################
# Second phase: load model and tokenizer from local
from transformers import T5Model, T5PreTrainedModel
from transformers import T5Config
import torch.nn as nn
from transformers.activations import NewGELUActivation
from typing import Optional, List
import torch
from transformers import T5Tokenizer
import pandas as pd
from chemistry_adapters.amino_acids import AminoAcidAdapter
import selfies as sf


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5ClassificationHead(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.dense_0 = nn.Linear(config.d_model, config.d_model // 4)
        self.norm_0 = T5LayerNorm(config.d_model // 4)
        self.relu_0 = NewGELUActivation()
        self.dropout_0 = nn.Dropout(p=config.classifier_dropout)

        self.dense_1 = nn.Linear(config.d_model // 4, config.d_model // 16)
        self.norm_1 = T5LayerNorm(config.d_model // 16)
        self.relu_1 = NewGELUActivation()
        self.dropout_1 = nn.Dropout(p=config.classifier_dropout)

        self.dense_2 = nn.Linear(config.d_model // 16, config.d_model // 64)
        self.norm_2 = T5LayerNorm(config.d_model // 64)
        self.relu_2 = NewGELUActivation()
        self.dropout_2 = nn.Dropout(p=config.classifier_dropout)

        self.out_proj = nn.Linear(config.d_model // 64, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout_0(
            self.relu_0(self.norm_0(self.dense_0(hidden_states)))
        )
        hidden_states = self.dropout_1(
            self.relu_1(self.norm_1(self.dense_1(hidden_states)))
        )
        hidden_states = self.dropout_2(
            self.relu_2(self.norm_2(self.dense_2(hidden_states)))
        )

        out = self.out_proj(hidden_states)

        return out


class T5ForSequenceClassification(T5PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.transformer = T5Model(config)
        self.classification_head = T5ClassificationHead(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(
            batch_size, -1, hidden_size
        )[:, -1, :]
        logits = self.classification_head(sentence_representation)

        return logits


class xBitterT5_predictor:
    def __init__(
        self,
        local_xBitterT5_640_ckpt,
        local_xBitterT5_720_ckpt,
        tokenizer_ckpt,
        device,
        hf_xBitterT5_640_ckpt="cbbl-skku-org/xBitterT5-640",
        hf_xBitterT5_720_ckpt="cbbl-skku-org/xBitterT5-720",
    ):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_ckpt)
        self.xBitterT5_640 = self.load_model(
            local_xBitterT5_640_ckpt, hf_xBitterT5_640_ckpt
        )
        self.xBitterT5_720 = self.load_model(
            local_xBitterT5_720_ckpt, hf_xBitterT5_720_ckpt
        )
        self.adapter = AminoAcidAdapter()

    def load_model(self, local_ckpt, hf_ckpt):
        model = T5ForSequenceClassification(T5Config.from_pretrained(hf_ckpt))
        state_dict = torch.load(local_ckpt)
        print(state_dict.keys())
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    def convert_sequence_to_smiles(self, sequence):
        return self.adapter.convert_amino_acid_sequence_to_smiles(sequence)

    def conver_smiles_to_selfies(self, smiles):
        return sf.encoder(smiles)

    def predict(self, input_dict, model_type="xBitterT5-720", batch_size=4):
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

        results = []
        if model_type == "xBitterT5-640":
            model = self.xBitterT5_640
        else:
            model = self.xBitterT5_720
        for i in range(0, len(text_inputs), batch_size):
            batch = text_inputs[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            predictions = torch.argmax(outputs, dim=1).tolist()
            probs = torch.softmax(outputs, dim=1).tolist()
            results.extend(
                [{"label": p, "score": prob[1]} for p, prob in zip(predictions, probs)]
            )

        y_pred, y_prob = [], []
        for pred in results:
            if pred["label"] == 1:
                y_prob.append(pred["score"])
                y_pred.append(1)
            else:
                y_prob.append(1 - pred["score"])
                y_pred.append(0)

        return {i: [y_prob[j], y_pred[j]] for j, i in enumerate(df["id"].tolist())}


# tokenizer = T5Tokenizer.from_pretrained("tokenizer")
# xBitterT5_640_dict = torch.load("model_640.pt")
# xBitterT5_720_dict = torch.load("model_720.pt")

# xBitterT5_640_ckpt = "cbbl-skku-org/xBitterT5-640"
# xBitterT5_720_ckpt = "cbbl-skku-org/xBitterT5-720"

# xBitterT5_640_config = T5Config.from_pretrained(xBitterT5_640_ckpt)
# xBitterT5_640 = T5ForSequenceClassification(xBitterT5_640_config)

# xBitterT5_720_config = T5Config.from_pretrained(xBitterT5_720_ckpt)
# xBitterT5_720 = T5ForSequenceClassification(xBitterT5_720_config)

# text = "This is a test"
# inputs_640 = tokenizer(text, return_tensors="pt")
# outputs_640 = xBitterT5_640(**inputs_640)
# print(outputs_640)

# inputs_720 = tokenizer(text, return_tensors="pt")
# outputs_720 = xBitterT5_720(**inputs_720)
# print(outputs_720)

if __name__ == "__main__":
    import time

    predictor = xBitterT5_predictor(
        tokenizer_ckpt="tokenizer",
        local_xBitterT5_640_ckpt="model_640.pt",
        local_xBitterT5_720_ckpt="model_720.pt",
        device="cpu",
    )
    input_dict = {1: "PA"}
    start = time.time()
    result = predictor.predict(input_dict)
    print(f"Time take in seconds: {time.time() - start}")
    print(result)
