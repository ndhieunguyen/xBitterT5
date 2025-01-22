import evaluate
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from torch.nn.functional import softmax
from torch import tensor
from sklearn.metrics import confusion_matrix, roc_curve, auc


bitter_metrics = evaluate.combine(
    ["accuracy", "f1", "precision", "recall", "matthews_correlation"]
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions[0], axis=1)
    prediction_scores = softmax(tensor(predictions[0]), dim=-1)
    prediction_scores = prediction_scores[:, 1].cpu().numpy()

    metrics = bitter_metrics.compute(predictions=preds, references=labels)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp)
    metrics.update(
        {
            "eval_specificity": specificity,
            "eval_tn": tn,
            "eval_fp": fp,
            "eval_fn": fn,
            "eval_tp": tp,
        }
    )

    fpr2, tpr2, _ = roc_curve(labels, prediction_scores, pos_label=1)
    auc2 = auc(fpr2, tpr2)
    metrics.update({"eval_auc": auc2})

    metrics = dict(sorted(metrics.items()))
    return metrics


def get_time_string():
    return datetime.now(tz=ZoneInfo("Asia/Seoul")).strftime("%Y_%m_%d__%H_%M_%S")
