from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from projetental.metrics import fb_cubed_score


def per_lemma_kmeans(df, vectors, oracle_k=False, kmin=2, kmax=10):
    results = defaultdict(list)

    for lemma in df["lemma"].unique():
        mask = df["lemma"] == lemma
        X, y_true = vectors[mask], df[mask]["sense_id"].values
        if len(X) < 2:
            continue

        if oracle_k:
            k = len(np.unique(y_true))
        else:
            best, best_score = None, -1
            for k in range(kmin, min(kmax, len(X)) + 1):
                km = KMeans(n_clusters=k, n_init="auto").fit(X)
                score = silhouette_score(X, km.labels_)
                if score > best_score:
                    best = km.labels_
                    best_score = score
            labels = best
        if oracle_k:
            km = KMeans(n_clusters=k, n_init="auto").fit(X)
            labels = km.labels_

        fb = fb_cubed_score(y_true, labels)
        for m, v in fb.items():
            results[m].append(v)

    return {m: np.mean(vs) for m, vs in results.items()}


def run_unsupervised(df, X, estimate_k=False, kmin=2, kmax=8):
    results = []

    for lemma in df["lemma"].unique():
        lemma_mask = df["lemma"] == lemma
        df_lemma = df[lemma_mask].reset_index(drop=True)
        X_lemma = X[lemma_mask]
        y_true = df_lemma["sense_id"].values

        if len(df_lemma) < 2:
            continue

        if estimate_k:

            best_k, best_score = 2, -1
            for k in range(kmin, min(kmax + 1, len(X_lemma))):
                try:
                    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
                    labels = kmeans.fit_predict(X_lemma)
                    score = silhouette_score(X_lemma, labels)
                    if score > best_score:
                        best_k, best_score = k, score
                except:
                    continue
            k = best_k
        else:

            k = len(np.unique(y_true))

        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(X_lemma)

        fb = fb_cubed_score(y_true, labels)

        results.append(
            {
                "Lemma": lemma,
                "K": k,
                "TrueK": len(np.unique(y_true)),
                "B-Cubed Precision": fb["B-Cubed Precision"],
                "B-Cubed Recall": fb["B-Cubed Recall"],
                "F-B-Cubed": fb["F-B-Cubed"],
                "N_instances": len(df_lemma),
                "Strategy": "estimated" if estimate_k else "oracle",
            }
        )

    return pd.DataFrame(results)
