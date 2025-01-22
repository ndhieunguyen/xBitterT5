from src.model import prepare_model, prepare_tokenizer
from src.data import prepare_dataset
from src.utils import compute_metrics, get_time_string
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
from datasets import load_from_disk
import os
import json


def main(args):
    set_seed(args.seed)
    print(f"Seed set to {args.seed}")

    os.environ["WANDB_PROJECT"] = f"Bitter_{os.path.basename(args.data_folder)}"

    if args.prepare_dataset:
        prepare_dataset(args)
        print("Dataset prepared")

    else:
        run_name = f"{'_'.join(args.chosen_features)}_{args.fold}_of_{args.k_folds}_{args.dropout}_{get_time_string()}"
        output_dir = os.path.join(
            args.output_dir,
            "_".join(args.chosen_features)
            + f"_{args.pretrained_name.split('/')[-1].replace('-', '_')}",
            "dropout_" + str(args.dropout),
            "fold_" + str(args.fold),
        )
        print(f"Save to output dir: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f)

        tokenizer = prepare_tokenizer(args)
        model = prepare_model(args)

        if args.k_folds == args.fold:
            data_path = os.path.join(
                args.data_folder,
                f"dataset_{'_'.join(args.chosen_features)}_{args.pretrained_name.split('/')[-1].replace('-', '_')}",
            )

        else:
            data_path = os.path.join(
                args.data_folder,
                f"fold_{args.fold}",
                f"dataset_{'_'.join(args.chosen_features)}_{args.pretrained_name.split('/')[-1].replace('-', '_')}",
            )
        dataset = load_from_disk(data_path)
        tokenized_dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
            ),
            batched=True,
        ).shuffle(args.seed)
        data_collator = DataCollatorWithPadding(tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            push_to_hub=args.push_to_hub,
            warmup_ratio=args.warmup_ratio,
            run_name=run_name,
            report_to=args.report_to,
            save_total_limit=1,
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=(
                tokenized_dataset["test"]
                if args.k_folds == args.fold
                else tokenized_dataset["val"]
            ),
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_dir)
        metrics = trainer.evaluate()
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    import argparse
    from config import config

    parser = argparse.ArgumentParser()
    for k, v in config.__dict__.items():
        if type(v) in [str, int, float]:
            parser.add_argument(f"--{k}", type=type(v), default=v)
        elif type(v) == bool:
            parser.add_argument(f"--{k}", action="store_false" if v else "store_true")
        elif type(v) == list:
            parser.add_argument(f"--{k}", nargs="*", type=type(v[0]), default=v)

    args = parser.parse_args()
    main(args)
