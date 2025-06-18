from typing import List, Tuple, Any
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from projetental.metrics import fb_cubed_score


def cop_kmeans(data, k, ml=[], cl=[], max_iter=100):
    try:

        X = np.array(data)
        n_samples = len(X)

        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(X)

        for _ in range(max_iter):

            ml_violations = 0
            for i, j in ml:
                if i < n_samples and j < n_samples and labels[i] != labels[j]:

                    cluster_i_size = np.sum(labels == labels[i])
                    cluster_j_size = np.sum(labels == labels[j])
                    if cluster_i_size >= cluster_j_size:
                        labels[j] = labels[i]
                    else:
                        labels[i] = labels[j]
                    ml_violations += 1

            cl_violations = 0
            for i, j in cl:
                if i < n_samples and j < n_samples and labels[i] == labels[j]:

                    available_clusters = [c for c in range(k) if c != labels[i]]
                    if available_clusters:
                        labels[j] = available_clusters[0]
                        cl_violations += 1

            if ml_violations == 0 and cl_violations == 0:
                break

        centers = []
        for cluster_id in range(k):
            cluster_points = X[labels == cluster_id]
            if len(cluster_points) > 0:
                centers.append(np.mean(cluster_points, axis=0))
            else:
                centers.append(np.zeros(X.shape[1]))

        return labels, np.array(centers)

    except Exception:
        return None, None


def generate_constraints(df_lemma, ratio=0.1):
    ml, cl = [], []
    n = len(df_lemma)

    for i in range(n):
        for j in range(i + 1, n):
            if df_lemma.iloc[i]["sense_id"] == df_lemma.iloc[j]["sense_id"]:
                ml.append((i, j))
            else:
                cl.append((i, j))

    return ml, cl


def run_cop_kmeans(
    df,
    get_vectors_fn,
    generate_constraints_fn,
    constraint_ratios=[0.1, 0.2],
    oracle_k=True,
    seed=0,
    verbose=True,
    kmin=2,
    kmax=8,
):
    results = []
    random.seed(seed)
    np.random.seed(seed)

    for ratio in constraint_ratios:
        for lemma in df["lemma"].unique():
            df_lemma = df[df["lemma"] == lemma].reset_index(drop=True)
            if len(df_lemma) < 2:
                continue

            X = get_vectors_fn(df_lemma)
            y_true = df_lemma["sense_id"].values
            k = (
                df_lemma["sense_id"].nunique()
                if oracle_k
                else estimate_k(X, kmin=kmin, kmax=kmax)
            )
            ml, cl = generate_constraints_fn(df_lemma)
            ml_subset = random.sample(ml, max(1, int(ratio * len(ml)))) if ml else []
            cl_subset = random.sample(cl, max(1, int(ratio * len(cl)))) if cl else []

            clusters, centers = cop_kmeans(X.tolist(), k, ml=ml_subset, cl=cl_subset)
            method = "COP-KMeans"
            if clusters is None:
                clusters = KMeans(
                    n_clusters=k, n_init="auto", random_state=seed
                ).fit_predict(X)
                method = "KMeans-Fallback"

            temp_df = pd.DataFrame({"sense_id": y_true, "pred_cluster": clusters})
            res = fb_cubed_score(y_true, clusters)
            results.append(
                {
                    "Lemma": lemma,
                    "ConstraintRatio": ratio,
                    "KStrategy": "oracle" if oracle_k else "estimated",
                    "B-Cubed Precision": res["B-Cubed Precision"],
                    "B-Cubed Recall": res["B-Cubed Recall"],
                    "F-B-Cubed": res["F-B-Cubed"],
                    "Method": method,
                }
            )

            if verbose:
                print(
                    f"{lemma:15} | BPrec: {res['B-Cubed Precision']:.4f} | BF1: {res['F-B-Cubed']:.4f} | {method}"
                )

    return results


def estimate_k(X, kmin=2, kmax=8):
    from sklearn.metrics import silhouette_score

    best_k, best_score = kmin, -1
    for k in range(kmin, min(kmax + 1, len(X))):
        try:
            labels = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k
