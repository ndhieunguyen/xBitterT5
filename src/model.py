# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import AutoTokenizer, T5Tokenizer, AutoConfig
from src.modeling_t5 import T5ForSequenceClassification


def prepare_tokenizer(args):
    try:
        try:
            return AutoTokenizer.from_pretrained(args.pretrained_name)
        except Exception as e:
            print(f"Error: {e}")
            return T5Tokenizer.from_pretrained(
                args.pretrained_name,
                do_lower_case=False,
            )
    except Exception as e:
        print(f"Error: {e}")
        return T5Tokenizer.from_pretrained(args.pretrained_name)


def check_unfreeze_layer(name, trainable_layers):
    flag = False
    for layer in trainable_layers:
        if name.startswith(f"transformer.decoder.block.{layer}"):
            flag = True
            break
    return flag


def prepare_model(args):
    id2lable = {0: "non-bitter", 1: "bitter"}
    label2id = {"non-bitter": 0, "bitter": 1}
    config = AutoConfig.from_pretrained(
        args.pretrained_name,
        cache_dir=args.cache_dir,
        num_labels=2,
        id2label=id2lable,
        label2id=label2id,
    )
    config.dropout_rate = args.dropout
    config.classifier_dropout = args.dropout
    config.problem_type = "single_label_classification"

    model = T5ForSequenceClassification.from_pretrained(
        args.pretrained_name,
        cache_dir=args.cache_dir,
        config=config,
    )
    model.to(args.accelerator)
    for name, param in model.named_parameters():
        if name.startswith("classification_head") or check_unfreeze_layer(
            name, args.trainable_layers
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model
