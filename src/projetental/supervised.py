from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from projetental.metrics import fb_cubed


def evaluate_classifier(clf, X, y, df, lemma_col="lemma", n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    fb_scores = []
    fb_precision_scores = []
    fb_recall_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        test_df = df.iloc[test_idx].copy()
        test_df["pred_cluster"] = y_pred
        fb = fb_cubed(
            test_df, pred_col="pred_cluster", gold_col="sense_id", lemma_col=lemma_col
        )
        fb_scores.append(fb["F-B-Cubed"])
        fb_precision_scores.append(fb["B-Cubed Precision"])
        fb_recall_scores.append(fb["B-Cubed Recall"])

    return {
        "f1_scores": f1_scores,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "fb_scores": fb_scores,
        "fb_precision_scores": fb_precision_scores,
        "fb_recall_scores": fb_recall_scores,
    }


def evaluate_mlp_classifier(X, y, df, lemma_col="lemma", n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f1_scores, precision_scores, recall_scores = [], [], []
    fb_scores, fb_precision_scores, fb_recall_scores = [], [], []

    clf = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=200,
        early_stopping=False,
        random_state=random_state,
    )

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1_scores.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        precision_scores.append(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        )
        recall_scores.append(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        )

        test_df = df.iloc[test_idx].copy()
        test_df["pred_cluster"] = y_pred
        fb = fb_cubed(
            test_df, pred_col="pred_cluster", gold_col="sense_id", lemma_col=lemma_col
        )
        fb_scores.append(fb["F-B-Cubed"])
        fb_precision_scores.append(fb["B-Cubed Precision"])
        fb_recall_scores.append(fb["B-Cubed Recall"])

    return {
        "f1_scores": f1_scores,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "fb_scores": fb_scores,
        "fb_precision_scores": fb_precision_scores,
        "fb_recall_scores": fb_recall_scores,
    }


def run_all_classifiers(
    df,
    get_vectors_func,
    embedding_types,
    k_values,
    n_splits=5,
):
    all_results = []
    for emb_type in embedding_types:
        print(f"\n=== Embedding: {emb_type} ===")
        for lemma, group in df.groupby("lemma"):
            X = get_vectors_func(group, emb_type)
            y = group["sense_id"].values

            if len(np.unique(y)) < 2:
                print(
                    f"Skipping lemma '{lemma}' for emb '{emb_type}' (only one class present)"
                )
                continue

            unique, counts = np.unique(y, return_counts=True)
            min_class_size = counts.min()
            curr_n_splits = min(n_splits, min_class_size)
            if curr_n_splits < 2:
                print(
                    f"Skipping lemma '{lemma}' for emb '{emb_type}' (too few samples for CV)"
                )
                continue

            scores = evaluate_classifier(
                LogisticRegression(max_iter=1000), X, y, group, n_splits=curr_n_splits
            )
            all_results.append(
                {
                    "Lemma": lemma,
                    "Embedding": emb_type,
                    "Classifier": "LogisticRegression",
                    "Mean F1": np.mean(scores["f1_scores"]),
                    "Mean Precision": np.mean(scores["precision_scores"]),
                    "Mean Recall": np.mean(scores["recall_scores"]),
                    "Mean F-B-Cubed": np.mean(scores["fb_scores"]),
                    "Mean B-Cubed Precision": np.mean(scores["fb_precision_scores"]),
                    "Mean B-Cubed Recall": np.mean(scores["fb_recall_scores"]),
                    "Best K for KNN": np.nan,
                    "Best KNN Mean F1": np.nan,
                }
            )

            scores = evaluate_classifier(SVC(), X, y, group, n_splits=curr_n_splits)
            all_results.append(
                {
                    "Lemma": lemma,
                    "Embedding": emb_type,
                    "Classifier": "SVM",
                    "Mean F1": np.mean(scores["f1_scores"]),
                    "Mean Precision": np.mean(scores["precision_scores"]),
                    "Mean Recall": np.mean(scores["recall_scores"]),
                    "Mean F-B-Cubed": np.mean(scores["fb_scores"]),
                    "Mean B-Cubed Precision": np.mean(scores["fb_precision_scores"]),
                    "Mean B-Cubed Recall": np.mean(scores["fb_recall_scores"]),
                    "Best K for KNN": np.nan,
                    "Best KNN Mean F1": np.nan,
                }
            )

            knn_f1s, knn_fbs, actual_ks = [], [], []
            for k in k_values:
                max_training_size = int((curr_n_splits - 1) / curr_n_splits * len(y))
                actual_k = min(k, max_training_size - 1)
                if actual_k < 1:
                    actual_k = 1
                if actual_k >= len(y):
                    continue
                scores = evaluate_classifier(
                    KNeighborsClassifier(n_neighbors=actual_k),
                    X,
                    y,
                    group,
                    n_splits=curr_n_splits,
                )
                knn_f1s.append(np.mean(scores["f1_scores"]))
                knn_fbs.append(np.mean(scores["fb_scores"]))
                actual_ks.append(actual_k)
                all_results.append(
                    {
                        "Lemma": lemma,
                        "Embedding": emb_type,
                        "Classifier": f"KNN-{actual_k}",
                        "Mean F1": np.mean(scores["f1_scores"]),
                        "Mean Precision": np.mean(scores["precision_scores"]),
                        "Mean Recall": np.mean(scores["recall_scores"]),
                        "Mean F-B-Cubed": np.mean(scores["fb_scores"]),
                        "Mean B-Cubed Precision": np.mean(
                            scores["fb_precision_scores"]
                        ),
                        "Mean B-Cubed Recall": np.mean(scores["fb_recall_scores"]),
                        "Best K for KNN": actual_k,
                        "Best KNN Mean F1": np.mean(scores["f1_scores"]),
                    }
                )

            if actual_ks:
                best_idx = int(np.argmax(knn_f1s))
                best_k = actual_ks[best_idx]
                best_f1 = knn_f1s[best_idx]
                best_fbc = knn_fbs[best_idx]
                all_results.append(
                    {
                        "Lemma": lemma,
                        "Embedding": emb_type,
                        "Classifier": "KNN-Best",
                        "Mean F1": best_f1,
                        "Mean Precision": np.nan,
                        "Mean Recall": np.nan,
                        "Mean F-B-Cubed": best_fbc,
                        "Mean B-Cubed Precision": np.nan,
                        "Mean B-Cubed Recall": np.nan,
                        "Best K for KNN": best_k,
                        "Best KNN Mean F1": best_f1,
                    }
                )

            scores = evaluate_mlp_classifier(X, y, group, n_splits=curr_n_splits)
            all_results.append(
                {
                    "Lemma": lemma,
                    "Embedding": emb_type,
                    "Classifier": "MLP",
                    "Mean F1": np.mean(scores["f1_scores"]),
                    "Mean Precision": np.mean(scores["precision_scores"]),
                    "Mean Recall": np.mean(scores["recall_scores"]),
                    "Mean F-B-Cubed": np.mean(scores["fb_scores"]),
                    "Mean B-Cubed Precision": np.mean(scores["fb_precision_scores"]),
                    "Mean B-Cubed Recall": np.mean(scores["fb_recall_scores"]),
                    "Best K for KNN": np.nan,
                    "Best KNN Mean F1": np.nan,
                }
            )

    results_df = pd.DataFrame(all_results)
    return results_df
