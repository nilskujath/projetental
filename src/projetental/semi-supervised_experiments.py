import pandas as pd
import warnings
from transformers import AutoTokenizer, AutoModel
from projetental.embeddings import get_camembert_embeddings
from projetental.semi_supervised import (
    generate_constraints,
    run_cop_kmeans,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")


df = pd.read_csv("data/processed_dataset.csv")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModel.from_pretrained("camembert-base")


def get_vectors_fn(df_lemma):
    return get_camembert_embeddings(df_lemma, tokenizer, model)


constraint_ratios = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
all_results = []

print("\n" + "=" * 60)
print("RUNNING COP-KMEANS EXPERIMENTS")
print("=" * 60)

for oracle_k in [True, False]:
    print(f"\n--- COP-KMeans with {'Oracle' if oracle_k else 'Estimated'} k ---")
    results = run_cop_kmeans(
        df,
        get_vectors_fn=get_vectors_fn,
        generate_constraints_fn=generate_constraints,
        constraint_ratios=constraint_ratios,
        oracle_k=oracle_k,
        seed=0,
        verbose=True,
    )
    all_results.extend(results)


results_df = pd.DataFrame(all_results)
results_df.to_csv("results/semisup_all_clustering_results.csv", index=False)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)


cop_kmeans_results = results_df[results_df["Method"].str.contains("COP-KMeans")]

if not cop_kmeans_results.empty:
    print("\n==== COP-KMeans Results (by Constraint Ratio and K Strategy) ====")
    cop_summary = (
        cop_kmeans_results.groupby(["ConstraintRatio", "KStrategy"])["F-B-Cubed"]
        .mean()
        .reset_index()
    )
    print(cop_summary)

print("\n==== Overall Best Results by Method ====")
method_summary = (
    results_df.groupby("Method")["F-B-Cubed"].agg(["mean", "std", "max"]).reset_index()
)
method_summary = method_summary.sort_values("mean", ascending=False)
print(method_summary)

print(f"\nResults saved to: results/semisup_all_clustering_results.csv")
