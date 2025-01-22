from transformers_interpret import SequenceClassificationExplainer
from typing import List, Tuple, Union
import torch


class xBitterT5_explainer(SequenceClassificationExplainer):
    def _make_input_reference_pair(
        self, text: Union[List, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if isinstance(text, list):
            raise NotImplementedError("Lists of text are not currently supported.")

        text_ids = self.encode(text)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # if no special tokens were added
        if len(text_ids) == len(input_ids):
            ref_input_ids = [self.ref_token_id] * len(text_ids)
        else:
            ref_input_ids = (
                [self.cls_token_id]
                + [self.ref_token_id] * len(text_ids)
                + [self.sep_token_id]
            )

        # Use this because pretrained BioT5 plus does not have cls_token_id
        ref_input_ids = [self.ref_token_id] * len(text_ids) + [self.sep_token_id]
        return (
            torch.tensor([input_ids], device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            len(text_ids),
        )
