import pandas as pd
import numpy as np
import warnings
from projetental.embeddings import get_vectors
from projetental.supervised import run_all_classifiers
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*least populated class.*")

df = pd.read_csv("data/processed_dataset.csv")


def get_vectors_func(group_df, embedding_type):
    if embedding_type.lower() == "tfidf":
        return get_vectors(group_df, "tfidf")
    elif embedding_type.lower() in ("camembert", "camembert", "camembert"):
        tokenizer = AutoTokenizer.from_pretrained("camembert-base")
        model = AutoModel.from_pretrained("camembert-base")
        return get_vectors(group_df, "camembert", tokenizer=tokenizer, model=model)
    elif embedding_type.lower() in ("camembert-large", "camembert-large"):
        tokenizer = AutoTokenizer.from_pretrained("camembert/camembert-large")
        model = AutoModel.from_pretrained("camembert/camembert-large")
        return get_vectors(
            group_df, "camembert-large", tokenizer=tokenizer, model=model
        )
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


embedding_types = ["tfidf", "camembert"]
k_values = [2, 4, 6, 8]

print("\n=== RUNNING COMPREHENSIVE SUPERVISED CLASSIFICATION EXPERIMENTS ===")

results_df = run_all_classifiers(
    df,
    get_vectors_func=get_vectors_func,
    embedding_types=embedding_types,
    k_values=k_values,
    n_splits=5,
)

results_df.to_csv("results/supervised_results.csv", index=False)

print("\n=== SUPERVISED EXPERIMENTS SUMMARY ===")
if len(results_df) > 0:
    print("Results by classifier and embedding (averaged across all lemmas):")
    summary = results_df.groupby(["Embedding", "Classifier"])[
        ["Mean F1", "Mean F-B-Cubed"]
    ].mean()
    print(summary)

    print(f"\nResults saved to: results/supervised_results.csv")
    print(f"Total lemmas processed: {len(results_df['Lemma'].unique())}")
    print(f"Total experiments run: {len(results_df)}")

    print("\nBest performing combinations:")
    best_f1 = results_df.loc[results_df["Mean F1"].idxmax()]
    best_fbc = results_df.loc[results_df["Mean F-B-Cubed"].idxmax()]
    print(
        f"Best F1: {best_f1['Embedding']} + {best_f1['Classifier']} = {best_f1['Mean F1']:.4f}"
    )
    print(
        f"Best F-B-Cubed: {best_fbc['Embedding']} + {best_fbc['Classifier']} = {best_fbc['Mean F-B-Cubed']:.4f}"
    )
else:
    print("No lemmas had sufficient data for supervised learning.")
    print("Results saved to: results/supervised_results.csv (empty)")
    print("Total lemmas processed: 0")
    print("Total experiments run: 0")
