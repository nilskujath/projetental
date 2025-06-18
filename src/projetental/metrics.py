from typing import Dict, Any
import numpy as np
import pandas as pd


def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score

    return {
        "Precision (Macro)": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall (Macro)": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1 (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision (Micro)": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "Recall (Micro)": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "F1 (Micro)": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


def fb_cubed(df, pred_col="pred_cluster", gold_col="sense_id", lemma_col="lemma"):
    precision_list = []
    recall_list = []

    for _, group in df.groupby(lemma_col):
        pred = group[pred_col].tolist()
        gold = group[gold_col].tolist()
        N = len(pred)

        for i in range(N):
            pred_cluster_i = pred[i]
            gold_sense_i = gold[i]

            same_pred = [j for j in range(N) if pred[j] == pred_cluster_i]
            same_gold = [j for j in range(N) if gold[j] == gold_sense_i]
            same_both = [j for j in same_pred if gold[j] == gold_sense_i]

            precision = len(same_both) / len(same_pred) if same_pred else 0.0
            recall = len(same_both) / len(same_gold) if same_gold else 0.0

            precision_list.append(precision)
            recall_list.append(recall)

    P = np.mean(precision_list)
    R = np.mean(recall_list)
    F = 2 * P * R / (P + R + 1e-8)
    return {"B-Cubed Precision": P, "B-Cubed Recall": R, "F-B-Cubed": F}


def fb_cubed_score(y_true, y_pred):
    temp_df = pd.DataFrame(
        {"sense_id": y_true, "pred_cluster": y_pred, "lemma": "temp"}
    )
    return fb_cubed(
        temp_df, pred_col="pred_cluster", gold_col="sense_id", lemma_col="lemma"
    )


def mapped_macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    from scipy.optimize import linear_sum_assignment

    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    y_pred_enc = LabelEncoder().fit_transform(y_pred)
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(col_ind, row_ind)}
    y_pred_mapped = [mapping.get(c, c) for c in y_pred_enc]
    return f1_score(y_true_enc, y_pred_mapped, average="macro")


def report_supervised(y_true, y_pred):
    metrics = classification_metrics(y_true, y_pred)
    metrics.update(fb_cubed_score(y_true, y_pred))
    return metrics


def report_clustering(y_true, y_pred):
    return fb_cubed_score(y_true, y_pred)
