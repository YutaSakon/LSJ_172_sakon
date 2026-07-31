from __future__ import annotations

import pandas as pd

#from function_new import add_uid_metrics, build_lm, standardize_numeric_columns


def _coerce_token_id(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_tokens(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return text.split()


def _find_spans(tokens: list[str], pattern: list[str]) -> list[tuple[int, int]]:
    if not tokens or not pattern or len(pattern) > len(tokens):
        return []

    lowered_tokens = [token.lower() for token in tokens]
    lowered_pattern = [token.lower() for token in pattern]
    max_start = len(tokens) - len(pattern) + 1

    spans = []
    for start in range(max_start):
        end = start + len(pattern)
        if lowered_tokens[start:end] == lowered_pattern:
            spans.append((start, end))
    return spans


def _find_ordered_matches(
    tokens: list[str],
    pattern: list[str],
) -> list[tuple[int, int]]:
    if not tokens or not pattern or len(pattern) > len(tokens):
        return []

    contiguous_spans = _find_spans(tokens, pattern)
    if contiguous_spans:
        return contiguous_spans

    lowered_tokens = [token.lower() for token in tokens]
    lowered_pattern = [token.lower() for token in pattern]
    matches: list[tuple[int, int]] = []

    for start in range(len(tokens)):
        if lowered_tokens[start] != lowered_pattern[0]:
            continue

        token_idx = start + 1
        pattern_idx = 1
        last_idx = start

        while token_idx < len(tokens) and pattern_idx < len(lowered_pattern):
            if lowered_tokens[token_idx] == lowered_pattern[pattern_idx]:
                last_idx = token_idx
                pattern_idx += 1
            token_idx += 1

        if pattern_idx == len(lowered_pattern):
            matches.append((start, last_idx + 1))

    return matches


def _find_space_insensitive_matches(
    tokens: list[str],
    pattern: list[str],
) -> list[tuple[int, int]]:
    if not tokens or not pattern:
        return []

    normalized_pattern = "".join(token.lower() for token in pattern)
    if not normalized_pattern:
        return []

    matches: list[tuple[int, int]] = []
    normalized_tokens = [token.lower() for token in tokens]

    for start in range(len(tokens)):
        combined = ""
        for end in range(start, len(tokens)):
            combined += normalized_tokens[end]
            if combined == normalized_pattern:
                matches.append((start, end + 1))
                break
            if len(combined) > len(normalized_pattern):
                break

    return matches


def _find_best_distance(row: pd.Series) -> float:
    subject_id = _coerce_token_id(row.get("subject_id"))
    sc_onset_id = _coerce_token_id(row.get("sc_onset_id"))
    if subject_id is not None and sc_onset_id is not None:
        return float(abs(subject_id - sc_onset_id))

    sentence_tokens = _as_tokens(row.get("sentence"))
    subject_tokens = _as_tokens(row.get("subject"))
    sc_onset_tokens = _as_tokens(row.get("sc_onset"))
    clause_tokens = _as_tokens(row.get("clause_text"))

    if not sentence_tokens or not sc_onset_tokens or not subject_tokens:
        return float("nan")

    sc_onset_spans = _find_ordered_matches(sentence_tokens, sc_onset_tokens)
    if not sc_onset_spans:
        sc_onset_spans = _find_space_insensitive_matches(
            sentence_tokens,
            sc_onset_tokens,
        )
    subject_spans = _find_ordered_matches(sentence_tokens, subject_tokens)
    if not subject_spans:
        subject_spans = _find_space_insensitive_matches(sentence_tokens, subject_tokens)
    clause_spans = _find_ordered_matches(sentence_tokens, clause_tokens)
    if not clause_spans:
        clause_spans = _find_space_insensitive_matches(sentence_tokens, clause_tokens)

    candidates: list[tuple[int, int]] = []

    for sc_onset_start, sc_onset_end in sc_onset_spans:
        for subject_start, subject_end in subject_spans:
            is_within_clause = True
            if clause_spans:
                is_within_clause = any(
                    clause_start <= sc_onset_start < clause_end
                    and clause_start <= subject_start
                    and subject_end <= clause_end
                    for clause_start, clause_end in clause_spans
                )

            if subject_start >= sc_onset_end:
                distance = subject_start - sc_onset_start
            elif subject_end <= sc_onset_start:
                distance = sc_onset_start - subject_start
            else:
                continue

            candidates.append((0 if is_within_clause else 1, distance))

    if candidates:
        candidates.sort()
        return float(candidates[0][1])

    return float("nan")


def add_distance_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    distance_column = "trigger_to_bahwa_subject_distance"
    z_column = f"{distance_column}_z"

    result[distance_column] = result.apply(_find_best_distance, axis=1)

    valid = result[distance_column].dropna()
    if valid.empty:
        result[z_column] = pd.NA
        return result

    mean = valid.mean()
    std = valid.std(ddof=0)

    if pd.isna(std) or std == 0:
        result[z_column] = result[distance_column].where(
            result[distance_column].isna(), 0.0
        )
    else:
        result[z_column] = (result[distance_column] - mean) / std

    return result


def _split_rows_with_empty_cells(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    empty_mask = df.isna().copy()
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            empty_mask[column] = empty_mask[column] | series.fillna("").astype(str).str.strip().eq("")

    rows_with_empty_cells = empty_mask.any(axis=1)
    valid_df = df.loc[~rows_with_empty_cells].copy()
    invalid_df = df.loc[rows_with_empty_cells].copy()
    return valid_df, invalid_df


def add_uid_metrics_after_duplicate_removal(
    input_csv_path: str,
    output_csv_path: str,
    standardized_output_csv_path: str | None = None,
    model_name: str = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-base",
    show_progress: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)

    required_columns = {"main_clause_text", "sc_onset"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"Input CSV is missing required columns for UID metrics: {missing_text}"
        )

    df, skipped_rows = _split_rows_with_empty_cells(df)
    if not skipped_rows.empty:
        display_df = skipped_rows.copy()
        display_df.insert(0, "original_row_number", skipped_rows.index + 2)
        print("Skipping rows with empty cells before adding UID metrics:")
        print(display_df.to_string(index=False))
        print(f"Skipped {len(skipped_rows)} row(s) with empty cells.")

    if df.empty:
        print("No rows remain after removing rows with empty cells.")
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        if standardized_output_csv_path is not None:
            standardize_numeric_columns(output_csv_path, standardized_output_csv_path)
        return df

    tokenizer, model, device = build_lm(model_name=model_name)
    result = add_uid_metrics(
        df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name=model_name,
        show_progress=show_progress,
    )
    result = add_distance_columns(result)

    result.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved UID- and distance-enriched CSV to: {output_csv_path}")

    if standardized_output_csv_path is not None and not result.empty:
        standardize_numeric_columns(output_csv_path, standardized_output_csv_path)

    return result


if __name__ == "__main__":
    input_csv_path = "/content/0516_LSJ_without_numbers.csv"
    output_csv_path = "/content/LSJ_uid.csv"
    standardized_output_csv_path = (
        "/content/LSJ_standardized.csv"
    )

    final_df = add_uid_metrics_after_duplicate_removal(
        input_csv_path=input_csv_path,
        output_csv_path=output_csv_path,
        standardized_output_csv_path=standardized_output_csv_path,
    )
    print(final_df.head().to_string(index=False))
