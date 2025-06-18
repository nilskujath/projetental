import pandas as pd
import numpy as np
import warnings
from projetental.embeddings import get_vectors
from projetental.unsupervised import run_unsupervised
from transformers import AutoTokenizer, AutoModel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")


df = pd.read_csv("data/processed_dataset.csv")
embedding_type = "camembert"
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModel.from_pretrained("camembert-base")


X = get_vectors(df, embedding_type, tokenizer=tokenizer, model=model)


oracle_df = run_unsupervised(df, X, estimate_k=False)
print("\n=== UNSUPERVISED (ORACLE k) RESULTS ===")
print(oracle_df[["Lemma", "F-B-Cubed"]].head())
print(f"Mean F-B-Cubed: {oracle_df['F-B-Cubed'].mean():.4f}")


est_df = run_unsupervised(df, X, estimate_k=True, kmin=2, kmax=8)
print("\n=== UNSUPERVISED (ESTIMATED k) RESULTS ===")
print(est_df[["Lemma", "F-B-Cubed"]].head())
print(f"Mean F-B-Cubed: {est_df['F-B-Cubed'].mean():.4f}")


oracle_df.to_csv("results/unsup_oracle.csv", index=False)
est_df.to_csv("results/unsup_estimated.csv", index=False)

print(f"\nResults saved to:")
print(f"  - results/unsup_oracle.csv")
print(f"  - results/unsup_estimated.csv")
