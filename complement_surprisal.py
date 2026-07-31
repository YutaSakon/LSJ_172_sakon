# @title
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
"""
from save_parsed_clause_csv_from_jsonl import (
    _find_word_by_id,
    _iter_jsonl_records,
)
"""
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


#DEFAULT_MODEL_NAME = "cahya/gpt2-large-indonesian-522M"
DEFAULT_MODEL_NAME = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-base"
BaseModule = nn.Module if nn is not None else object


@dataclass
class PCAProjector:
    mean: np.ndarray
    components: np.ndarray


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray


class ComplementPresenceClassifier(BaseModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int, int] = (128, 64, 32),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                continue
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


def _iter_with_progress(items, show_progress: bool, desc: str):
    if not show_progress:
        return items
    if tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def _normalize_surface(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).lower().strip()


def _clean_display_text(text: str) -> str:
    cleaned = re.sub(r"\S*\d\S*", "", str(text))
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _safe_surprisal(probability: float) -> float:
    if pd.isna(probability):
        return math.nan
    if probability <= 0.0:
        return math.inf
    return float(-math.log2(probability))


def _z_score_series(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid_values = numeric_values.dropna()
    if valid_values.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)

    mean = valid_values.mean()
    std = valid_values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return numeric_values.where(numeric_values.isna(), 0.0)

    return (numeric_values - mean) / std


def _resolve_surface_series(df: pd.DataFrame) -> pd.Series:
    if "target_surface" in df.columns:
        return df["target_surface"]
    if "trigger" in df.columns:
        return df["trigger"]
    raise ValueError(
        "Input CSV must contain either 'target_surface' or 'trigger'."
    )


def _series_is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return False
    return bool(np.all(np.isclose(valid, np.round(valid), atol=1e-6)))


def _resolve_occurrence_key_columns(df: pd.DataFrame) -> list[str]:
    preferred_keys: list[list[str]] = []
    if "trigger_id" in df.columns and _series_is_integer_like(df["trigger_id"]):
        preferred_keys.extend(
            [
                ["source_file", "sentence_id", "trigger_id", "target_surface"],
                ["source_file", "sentence_id", "trigger_id", "trigger"],
            ]
        )
    if "words_before_trigger" in df.columns and _series_is_integer_like(
        df["words_before_trigger"]
    ):
        preferred_keys.extend(
            [
                ["source_file", "sentence_id", "words_before_trigger", "target_surface"],
                ["source_file", "sentence_id", "words_before_trigger", "trigger"],
            ]
        )
    preferred_keys.extend(
        [
            ["source_file", "sentence_id", "trigger", "target_surface"],
            ["source_file", "sentence_id", "trigger"],
        ]
    )
    for candidate in preferred_keys:
        if all(column in df.columns for column in candidate):
            return candidate

    raise ValueError(
        "Input CSV must contain either "
        "integer-like ('source_file', 'sentence_id', 'trigger_id', 'target_surface') "
        "or ('source_file', 'sentence_id', 'words_before_trigger', 'target_surface') "
        "or ('source_file', 'sentence_id', 'trigger', 'target_surface')."
    )


def _prepare_positive_examples(input_csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    positive_df = pd.read_csv(input_csv_path).copy()
    positive_df["target_surface"] = _resolve_surface_series(positive_df).map(
        _normalize_surface
    )
    positive_df = positive_df[positive_df["target_surface"] != ""].copy()
    positive_df["complement_present"] = 1

    key_columns = _resolve_occurrence_key_columns(positive_df)
    positive_df = _sort_like_save_parsed_clause_csv_from_jsonl(positive_df)
    occurrence_counts = (
        positive_df.groupby(key_columns, dropna=False)
        .size()
        .rename("complement_clause_count")
        .reset_index()
    )
    positive_df = positive_df.drop_duplicates(subset=key_columns, keep="first").copy()
    positive_df = positive_df.merge(
        occurrence_counts,
        on=key_columns,
        how="left",
    )
    return positive_df.reset_index(drop=True), key_columns


def _sort_like_save_parsed_clause_csv_from_jsonl(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ["verb_rank", "source_file", "sentence_id", "trigger_id", "clause_head_id"]
        if column in df.columns
    ]
    if not sort_columns:
        return df.reset_index(drop=True)
    return df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _build_occurrence_key_from_values(
    values: dict[str, object] | pd.Series,
    key_columns: list[str],
) -> tuple[object, ...]:
    return tuple(values.get(column) for column in key_columns)


def _build_negative_row_template(
    record: dict,
    word: dict,
    target_surface: str,
    template_columns: list[str],
) -> dict[str, object]:
    row = {column: pd.NA for column in template_columns}
    word_id = word["id"]
    next_word = _find_word_by_id(record.get("words", []), word_id + 1)
    marker_type = "No"
    if next_word and (next_word.get("text") or "").lower() == "bahwa":
        marker_type = "bahwa"
    elif next_word and next_word.get("text") == ",":
        marker_type = "comma"

    values = {
        "source_file": record.get("source_file"),
        "sentence_id": record.get("sentence_id"),
        "sentence": record.get("sentence"),
        "trigger": word.get("text"),
        "target_surface": target_surface,
        "trigger_lemma": _normalize_surface(word.get("lemma")),
        "trigger_id": word_id,
        "marker_type": marker_type,
        "has_bahwa": int(marker_type == "bahwa"),
        "words_before_trigger": len(
            [
                candidate
                for candidate in record.get("words", [])
                if isinstance(candidate.get("id"), int) and candidate["id"] < word_id
            ]
        ),
        "complement_present": 0,
        "complement_clause_count": 0,
    }
    row.update(values)
    return row


def _collect_target_occurrences_from_jsonl(
    parsed_jsonl_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    target_surfaces: set[str],
    template_columns: list[str],
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    if not target_surfaces:
        return pd.DataFrame(columns=template_columns)

    rows: list[dict[str, object]] = []

    sentence_iter = _iter_jsonl_records(
        parsed_jsonl_path,
        encoding=encoding,
        show_progress=show_progress,
        desc_prefix="Collecting non-CSV target occurrences",
    )

    for record in sentence_iter:
        words = record.get("words", [])
        for word in words:
            word_id = word.get("id")
            if not isinstance(word_id, int):
                continue

            target_surface = _normalize_surface(word.get("text"))
            if target_surface not in target_surfaces:
                continue

            row = _build_negative_row_template(
                record=record,
                word=word,
                target_surface=target_surface,
                template_columns=template_columns,
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=template_columns)

    return pd.DataFrame(rows, columns=template_columns)


def collect_negative_examples_from_jsonl(
    parsed_jsonl_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    positive_df: pd.DataFrame,
    key_columns: list[str],
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    target_surfaces = set(positive_df["target_surface"].dropna().tolist())
    if not target_surfaces:
        return positive_df.iloc[0:0].copy()

    positive_keys = {
        _build_occurrence_key_from_values(row, key_columns)
        for _, row in positive_df.iterrows()
    }
    all_occurrences_df = _collect_target_occurrences_from_jsonl(
        parsed_jsonl_path=parsed_jsonl_path,
        target_surfaces=target_surfaces,
        template_columns=positive_df.columns.tolist(),
        encoding=encoding,
        show_progress=show_progress,
    )
    if all_occurrences_df.empty:
        return positive_df.iloc[0:0].copy()

    negative_mask = [
        _build_occurrence_key_from_values(row, key_columns) not in positive_keys
        for _, row in all_occurrences_df.iterrows()
    ]
    negative_df = all_occurrences_df.loc[negative_mask].copy()
    if negative_df.empty:
        return positive_df.iloc[0:0].copy()

    return _sort_like_save_parsed_clause_csv_from_jsonl(negative_df).reset_index(drop=True)


def _build_sentence_key(values: dict[str, object] | pd.Series) -> tuple[object, object]:
    return values.get("source_file"), values.get("sentence_id")


def _load_required_records(
    parsed_jsonl_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    required_sentence_keys: set[tuple[object, object]],
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> dict[tuple[object, object], dict[str, Any]]:
    records: dict[tuple[object, object], dict[str, Any]] = {}
    if not required_sentence_keys:
        return records

    sentence_iter = _iter_jsonl_records(
        parsed_jsonl_path,
        encoding=encoding,
        show_progress=show_progress,
        desc_prefix="Loading sentence records for embeddings",
    )
    for record in sentence_iter:
        sentence_key = (record.get("source_file"), record.get("sentence_id"))
        if sentence_key in required_sentence_keys:
            records[sentence_key] = record

    return records


def _coerce_int(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isclose(value, round(float(value)), atol=1e-6):
            return None
        return int(round(float(value)))
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isclose(float_value, round(float_value), atol=1e-6):
        return None
    return int(round(float_value))


def _find_trigger_id_for_row(row: pd.Series, record: dict[str, Any]) -> int | None:
    direct_trigger_id = _coerce_int(row.get("trigger_id"))
    words = record.get("words", [])
    words_by_id = {
        word["id"]: word for word in words if isinstance(word.get("id"), int)
    }
    if direct_trigger_id is not None and direct_trigger_id in words_by_id:
        return direct_trigger_id

    trigger_surface = _normalize_surface(row.get("trigger"))
    if not trigger_surface:
        trigger_surface = _normalize_surface(row.get("target_surface"))

    candidates = [
        word["id"]
        for word in words
        if isinstance(word.get("id"), int)
        and _normalize_surface(word.get("text")) == trigger_surface
    ]
    if not candidates:
        return None

    words_before_trigger = _coerce_int(row.get("words_before_trigger"))
    if words_before_trigger is not None:
        for candidate_id in candidates:
            candidate_words_before = len(
                [
                    word
                    for word in words
                    if isinstance(word.get("id"), int) and word["id"] < candidate_id
                ]
            )
            if candidate_words_before == words_before_trigger:
                return candidate_id

    return candidates[0]


def _find_prefix_end_id_for_row(
    row: pd.Series,
    record: dict[str, Any],
    trigger_id: int,
) -> int:
    return trigger_id + 1


def _build_prefix_text(record: dict[str, Any], prefix_end_id: int) -> str:
    words = record.get("words", [])
    ordered_tokens = [
        word.get("text", "")
        for word in sorted(
            (
                word
                for word in words
                if isinstance(word.get("id"), int) and word["id"] < prefix_end_id
            ),
            key=lambda item: item["id"],
        )
    ]
    return _clean_display_text(" ".join(token for token in ordered_tokens if token).strip())


def _load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
):
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required to build complement surprisal features."
        )
    if torch is None:
        raise ImportError("torch is required to build complement surprisal features.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device}: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if device == "cpu":
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    # NumPy cannot directly consume bfloat16 tensors, so cast in torch first.
    return tensor.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)


def add_prefix_context_columns(
    df: pd.DataFrame,
    parsed_jsonl_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        result["context_prefix_text"] = pd.Series(dtype=str)
        result["context_found_in_jsonl"] = pd.Series(dtype="Int64")
        return result

    required_sentence_keys = {
        _build_sentence_key(row)
        for _, row in result.iterrows()
    }
    record_map = _load_required_records(
        parsed_jsonl_path=parsed_jsonl_path,
        required_sentence_keys=required_sentence_keys,
        encoding=encoding,
        show_progress=show_progress,
    )

    prefix_texts = []
    found_flags = []
    row_iter = _iter_with_progress(
        list(result.iterrows()),
        show_progress=show_progress,
        desc="Building pre-onset contexts",
    )

    for _, row in row_iter:
        record = record_map.get(_build_sentence_key(row))
        if record is None:
            prefix_texts.append(pd.NA)
            found_flags.append(0)
            continue

        trigger_id = _find_trigger_id_for_row(row, record)
        if trigger_id is None:
            prefix_texts.append(pd.NA)
            found_flags.append(0)
            continue

        prefix_end_id = _find_prefix_end_id_for_row(row, record, trigger_id)
        prefix_text = _build_prefix_text(record, prefix_end_id)
        prefix_texts.append(prefix_text if prefix_text else pd.NA)
        found_flags.append(int(bool(prefix_text)))

    result["context_prefix_text"] = [
        _clean_display_text(value) if not pd.isna(value) else pd.NA
        for value in prefix_texts
    ]
    result["context_found_in_jsonl"] = pd.Series(found_flags, dtype="Int64")
    return result


def _get_trigger_char_span(prefix_text: str) -> tuple[int, int] | None:
    normalized_text = str(prefix_text).rstrip()
    if not normalized_text:
        return None
    start = normalized_text.rfind(" ") + 1
    end = len(normalized_text)
    if start >= end:
        return None
    return start, end


def _get_trigger_span_embedding(prefix_text: str, tokenizer, model, device) -> np.ndarray:
    encoded = tokenizer(
        prefix_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_state = outputs.hidden_states[-1][0]
    span = _get_trigger_char_span(prefix_text)
    if span is None:
        return _tensor_to_numpy_float32(hidden_state[-1])

    span_start, span_end = span
    span_indices = []
    for idx, (start, end) in enumerate(offsets):
        if (start, end) == (0, 0):
            continue
        if not (end <= span_start or span_end <= start):
            span_indices.append(idx)

    if not span_indices:
        return _tensor_to_numpy_float32(hidden_state[-1])

    span_embedding = hidden_state[span_indices].mean(dim=0)
    return _tensor_to_numpy_float32(span_embedding)


def extract_context_embeddings(
    df: pd.DataFrame,
    tokenizer,
    model,
    device,
    show_progress: bool = True,
) -> np.ndarray:
    if torch is None:
        raise ImportError("torch is required to compute contextual embeddings.")

    prefixes = df["context_prefix_text"].tolist()
    embeddings: list[np.ndarray] = []
    cache: dict[str, np.ndarray] = {}
    row_iter = _iter_with_progress(
        prefixes,
        show_progress=show_progress,
        desc="Extracting LM context embeddings",
    )

    embedding_dim: int | None = None
    for prefix_text in row_iter:
        if pd.isna(prefix_text) or not str(prefix_text).strip():
            if embedding_dim is None:
                embedding_dim = (
                    int(getattr(model.config, "hidden_size", 0))
                    or int(getattr(model.config, "n_embd", 0))
                    or 1
                )
            embeddings.append(np.full(embedding_dim, np.nan, dtype=np.float32))
            continue

        prefix_text = str(prefix_text)
        if prefix_text not in cache:
            cache[prefix_text] = _get_trigger_span_embedding(
                prefix_text,
                tokenizer,
                model,
                device,
            )
            if embedding_dim is None:
                embedding_dim = cache[prefix_text].shape[0]
        embeddings.append(cache[prefix_text])

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack(embeddings).astype(np.float32, copy=False)


def _fit_pca(train_matrix: np.ndarray, n_components: int) -> PCAProjector:
    effective_components = max(
        1,
        min(n_components, train_matrix.shape[0], train_matrix.shape[1]),
    )
    mean = train_matrix.mean(axis=0, keepdims=True)
    centered = train_matrix - mean
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:effective_components]
    components = eigenvectors[:, order]
    return PCAProjector(mean=mean.squeeze(0), components=components)


def _transform_pca(matrix: np.ndarray, projector: PCAProjector) -> np.ndarray:
    centered = matrix - projector.mean
    return centered @ projector.components


def _fit_standardizer(train_matrix: np.ndarray) -> Standardizer:
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return Standardizer(mean=mean, std=std)


def _transform_standardized(
    matrix: np.ndarray,
    standardizer: Standardizer,
) -> np.ndarray:
    return (matrix - standardizer.mean) / standardizer.std


def _split_train_validation_indices(
    labels: np.ndarray,
    validation_ratio: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    all_indices = np.arange(len(labels))
    positive_indices = all_indices[labels == 1]
    negative_indices = all_indices[labels == 0]

    def _take_validation(indices: np.ndarray) -> np.ndarray:
        if len(indices) <= 1:
            return np.array([], dtype=int)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        val_size = max(1, int(round(len(indices) * validation_ratio)))
        val_size = min(val_size, len(indices) - 1)
        return shuffled[:val_size]

    validation_indices = np.concatenate(
        [_take_validation(positive_indices), _take_validation(negative_indices)]
    )
    validation_indices = np.unique(validation_indices)
    validation_index_set = set(validation_indices.tolist())
    training_indices = np.array(
        [idx for idx in all_indices if idx not in validation_index_set],
        dtype=int,
    )

    if len(training_indices) == 0:
        training_indices = all_indices
        validation_indices = all_indices

    if len(validation_indices) == 0:
        validation_indices = training_indices

    return training_indices, validation_indices


def train_complement_presence_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    random_seed: int = 42,
    learning_rate: float = 2e-3,
    batch_size: int = 64,
    max_epochs: int = 60,
    patience: int = 8,
) -> ComplementPresenceClassifier:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError("torch is required to train the complement classifier.")
    if len(train_features) == 0:
        raise ValueError("At least one training example is required.")

    train_x = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    valid_x = torch.tensor(validation_features, dtype=torch.float32)
    valid_y = torch.tensor(validation_labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ComplementPresenceClassifier(input_dim=train_features.shape[1]).to(device)
    positive_count = int((train_labels == 1).sum())
    negative_count = int((train_labels == 0).sum())
    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1)],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_validation_loss = math.inf
    epochs_without_improvement = 0

    for _ in range(max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(valid_x.to(device))
            validation_loss = criterion(
                validation_logits,
                valid_y.to(device),
            ).item()

        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_complement_logits(
    model: ComplementPresenceClassifier,
    features: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    if torch is None or DataLoader is None or TensorDataset is None:
        raise ImportError("torch is required to predict complement probabilities.")

    device = next(model.parameters()).device
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=False,
    )

    logits = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_logits = model(batch_x.to(device)).squeeze(1).cpu().numpy()
            logits.append(batch_logits)

    if not logits:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(logits).astype(np.float32, copy=False)


def _fit_temperature_scaling(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    device: str,
    max_iter: int = 2000,
    learning_rate: float = 0.01,
) -> float:
    if len(validation_logits) == 0 or np.unique(validation_labels).size < 2:
        return 1.0

    logits_tensor = torch.tensor(validation_logits, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(
        validation_labels.astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    log_temperature = torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([log_temperature], lr=learning_rate)

    for _ in range(max_iter):
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature) + 1e-6
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits_tensor / temperature,
            labels_tensor,
        )
        loss.backward()
        optimizer.step()

    return float((torch.exp(log_temperature) + 1e-6).detach().cpu().item())


def predict_complement_probabilities(
    model: ComplementPresenceClassifier,
    features: np.ndarray,
    temperature: float = 1.0,
    batch_size: int = 256,
) -> np.ndarray:
    logits = predict_complement_logits(
        model=model,
        features=features,
        batch_size=batch_size,
    )
    if logits.size == 0:
        return logits
    scaled_logits = logits / max(float(temperature), 1e-6)
    return (1.0 / (1.0 + np.exp(-scaled_logits))).astype(np.float32, copy=False)


def add_model_based_surprisal_columns(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    pca_components: int = 50,
    random_seed: int = 42,
) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        result["target_verb_total_count"] = pd.Series(dtype="Int64")
        result["complement_taking_count"] = pd.Series(dtype="Int64")
        result["complement_taking_ratio"] = pd.Series(dtype=float)
        result["complement_probability"] = pd.Series(dtype=float)
        result["complement_present_surprisal"] = pd.Series(dtype=float)
        result["complement_present_surprisal_z"] = pd.Series(dtype=float)
        result["complement_absent_surprisal"] = pd.Series(dtype=float)
        result["observed_complement_event_probability"] = pd.Series(dtype=float)
        result["observed_complement_event_surprisal"] = pd.Series(dtype=float)
        result["observed_complement_event_surprisal_z"] = pd.Series(dtype=float)
        result["temperature_scaling_T"] = pd.Series(dtype=float)
        return result

    result["complement_present"] = (
        pd.to_numeric(result["complement_present"], errors="coerce")
        .fillna(0)
        .astype("Int64")
    )

    stats = (
        result.groupby("target_surface", dropna=False)["complement_present"]
        .agg(target_verb_total_count="size", complement_taking_count="sum")
        .reset_index()
    )
    stats["complement_taking_ratio"] = (
        stats["complement_taking_count"] / stats["target_verb_total_count"]
    )
    stats["complement_taking_ratio_z"] = _z_score_series(
        stats["complement_taking_ratio"]
    )
    overlap_columns = [
        column
        for column in stats.columns
        if column in result.columns and column != "target_surface"
    ]
    if overlap_columns:
        result = result.drop(columns=overlap_columns)
    result = result.merge(stats, on="target_surface", how="left")

    valid_mask = ~np.isnan(embeddings).any(axis=1)
    probabilities = np.full(len(result), np.nan, dtype=np.float32)
    temperature_values = np.full(len(result), np.nan, dtype=np.float32)
    labels = result["complement_present"].to_numpy(dtype=np.int64)

    if valid_mask.any():
        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]
        valid_positions = np.flatnonzero(valid_mask)

        if np.unique(valid_labels).size < 2:
            probabilities[valid_mask] = float(valid_labels.mean())
            temperature_values[valid_mask] = 1.0
        else:
            training_indices, validation_indices = _split_train_validation_indices(
                valid_labels,
                validation_ratio=0.2,
                random_seed=random_seed,
            )
            if len(training_indices) == 0:
                training_indices = np.arange(len(valid_labels))
            if len(validation_indices) == 0:
                validation_indices = training_indices

            train_embeddings = valid_embeddings[training_indices]
            validation_embeddings = valid_embeddings[validation_indices]
            standardizer = _fit_standardizer(train_embeddings)
            train_standardized = _transform_standardized(
                train_embeddings,
                standardizer,
            )
            validation_standardized = _transform_standardized(
                validation_embeddings,
                standardizer,
            )
            all_standardized = _transform_standardized(
                valid_embeddings,
                standardizer,
            )
            projector = _fit_pca(train_standardized, n_components=pca_components)
            train_features = _transform_pca(
                train_standardized,
                projector,
            ).astype(np.float32, copy=False)
            validation_features = _transform_pca(
                validation_standardized,
                projector,
            ).astype(np.float32, copy=False)
            all_features = _transform_pca(
                all_standardized,
                projector,
            ).astype(np.float32, copy=False)

            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            classifier = train_complement_presence_classifier(
                train_features=train_features,
                train_labels=valid_labels[training_indices],
                validation_features=validation_features,
                validation_labels=valid_labels[validation_indices],
                random_seed=random_seed,
            )
            validation_logits = predict_complement_logits(
                classifier,
                validation_features,
            )
            calibration_temperature = _fit_temperature_scaling(
                validation_logits=validation_logits,
                validation_labels=valid_labels[validation_indices],
                device=next(classifier.parameters()).device.type,
            )
            probabilities[valid_positions] = predict_complement_probabilities(
                classifier,
                all_features,
                temperature=calibration_temperature,
            )
            temperature_values[valid_positions] = calibration_temperature

    probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
    result["complement_probability"] = probabilities
    result["complement_present_surprisal"] = result["complement_probability"].map(
        _safe_surprisal
    )
    complement_present_surprisal_for_z = result["complement_present_surprisal"].where(
        result["complement_present"] == 1
    )
    result["complement_present_surprisal_z"] = _z_score_series(
        complement_present_surprisal_for_z
    )
    result["complement_absent_surprisal"] = (
        1.0 - result["complement_probability"]
    ).map(_safe_surprisal)

    observed_probabilities = np.where(
        result["complement_present"].to_numpy(dtype=np.int64) == 1,
        probabilities,
        1.0 - probabilities,
    )
    result["observed_complement_event_probability"] = observed_probabilities
    result["observed_complement_event_surprisal"] = [
        _safe_surprisal(probability) for probability in observed_probabilities
    ]
    result["observed_complement_event_surprisal_z"] = _z_score_series(
        result["observed_complement_event_surprisal"]
    )
    result["temperature_scaling_T"] = temperature_values

    for column in ("target_verb_total_count", "complement_taking_count"):
        result[column] = result[column].astype("Int64")

    return result


def build_complement_presence_surprisal_dataset(
    input_csv_path: str,
    parsed_jsonl_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    output_csv_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    pca_components: int = 50,
    random_seed: int = 42,
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positive_df, key_columns = _prepare_positive_examples(input_csv_path)
    if positive_df.empty:
        empty_df = add_model_based_surprisal_columns(
            positive_df,
            embeddings=np.empty((0, 0), dtype=np.float32),
            pca_components=pca_components,
            random_seed=random_seed,
        )
        empty_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"No target surfaces were found. Saved empty CSV to: {output_path}")
        return empty_df

    negative_df = collect_negative_examples_from_jsonl(
        parsed_jsonl_path=parsed_jsonl_path,
        positive_df=positive_df,
        key_columns=key_columns,
        encoding=encoding,
        show_progress=show_progress,
    )
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True, sort=False)
    combined_df = _sort_like_save_parsed_clause_csv_from_jsonl(combined_df)
    combined_df = add_prefix_context_columns(
        combined_df,
        parsed_jsonl_path=parsed_jsonl_path,
        encoding=encoding,
        show_progress=show_progress,
    )

    tokenizer, model, device = _load_embedding_model(model_name=model_name)
    embeddings = extract_context_embeddings(
        combined_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        show_progress=show_progress,
    )
    result = add_model_based_surprisal_columns(
        combined_df,
        embeddings=embeddings,
        pca_components=pca_components,
        random_seed=random_seed,
    )
    result = result[result["complement_present"] == 1].reset_index(drop=True)

    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved complement-presence surprisal CSV to: {output_path}")
    return result


if __name__ == "__main__":
    input_csv_path = "/content/LSJ_standardized.csv"
    parsed_jsonl_path = "/content/ind_parsed.jsonl"
    output_csv_path = "/content/LSJ_final.csv"

    final_df = build_complement_presence_surprisal_dataset(
        input_csv_path=input_csv_path,
        parsed_jsonl_path=parsed_jsonl_path,
        output_csv_path=output_csv_path,
    )
    print(final_df.head().to_string(index=False))
