import numpy as np
import pandas as pd
from typing import Optional, List, Union
from transformers import AutoTokenizer, AutoModel


def get_tfidf_vectors(df: pd.DataFrame, window_size: int = 5):
    from sklearn.feature_extraction.text import TfidfVectorizer

    contexts = []
    for _, row in df.iterrows():
        tokens = row["sentence"].split()
        i = row["target_index"]
        start, end = max(0, i - window_size), min(len(tokens), i + window_size + 1)
        context = [tokens[j] for j in range(start, end) if j != i]
        contexts.append(" ".join(context))
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(contexts).toarray()


def get_fasttext_model(lang: str = "fr"):
    import fasttext
    import fasttext.util

    fasttext.util.download_model(lang, if_exists="ignore")
    return fasttext.load_model(f"cc.{lang}.300.bin")


def get_fasttext_embeddings(
    sentence: str, target_index: int, ft_model, window_size: int = 5
):
    tokens = sentence.split()
    target_word = tokens[target_index]
    target_vector = ft_model.get_word_vector(target_word.lower())
    start = max(0, target_index - window_size)
    end = min(len(tokens), target_index + window_size + 1)
    context = [
        tokens[i]
        for i in range(start, end)
        if i != target_index and tokens[i].isalpha()
    ]
    context_vecs = [ft_model.get_word_vector(w.lower()) for w in context]
    if context_vecs:
        context_avg = np.mean(context_vecs, axis=0)
        return target_vector + context_avg
    else:
        return target_vector


def get_camembert_embeddings(
    df: pd.DataFrame, tokenizer, model, aggregate: str = "token"
):
    import torch

    vectors = []
    for _, row in df.iterrows():
        sentence = row["sentence"]
        target_index = row["target_index"]
        words = sentence.split()

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )

        model_inputs = {
            k: v for k, v in encoding.items() if k in ["input_ids", "attention_mask"]
        }

        with torch.no_grad():
            output = model(**model_inputs)

        if aggregate == "token":
            word_ids = encoding.word_ids()
            token_indices = [
                i for i, w_id in enumerate(word_ids) if w_id == target_index
            ]
            if token_indices:
                vec = (
                    output.last_hidden_state[0, token_indices, :]
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )
            else:
                vec = output.last_hidden_state[0, 0, :].cpu().numpy()
        else:
            vec = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        vectors.append(vec)

    return np.stack(vectors)


def get_vectors(
    df: pd.DataFrame,
    embedding_type: str,
    ft_model=None,
    tokenizer=None,
    model=None,
    window_size: int = 5,
) -> np.ndarray:

    embedding_type = embedding_type.lower()
    if embedding_type == "tfidf":
        return get_tfidf_vectors(df, window_size=window_size)
    elif embedding_type == "fasttext":
        if ft_model is None:
            raise ValueError(
                "FastText model must be provided for 'fasttext' embeddings."
            )
        return get_fasttext_embeddings(df, ft_model, window_size=window_size)
    elif embedding_type in (
        "camembert",
        "camembert-large",
        "camembert",
        "camembert-large",
    ):
        if tokenizer is None or model is None:
            from transformers import AutoTokenizer, AutoModel

            if embedding_type in ("camembert", "camembert"):
                tokenizer = AutoTokenizer.from_pretrained("camembert-base")
                model = AutoModel.from_pretrained("camembert-base")
            elif embedding_type in ("camembert-large", "camembert-large"):
                tokenizer = AutoTokenizer.from_pretrained("camembert/camembert-large")
                model = AutoModel.from_pretrained("camembert/camembert-large")
        return get_camembert_embeddings(df, tokenizer, model)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
