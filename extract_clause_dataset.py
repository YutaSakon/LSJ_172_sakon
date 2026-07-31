#JSONLから抽出
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _iter_with_progress(items, show_progress: bool, desc: str):
    if not show_progress:
        return items

    if tqdm is not None:
        return tqdm(items, desc=desc)

    return items


def _normalize_paths(
    file_path: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> list[Path]:
    if isinstance(file_path, (str, Path)):
        return [Path(file_path)]
    return [Path(path) for path in file_path]


def _iter_jsonl_records(
    file_paths: str | Path | list[str | Path] | tuple[str | Path, ...],
    encoding: str = "utf-8",
    show_progress: bool = True,
    desc_prefix: str = "Reading parsed sentences",
):
    for current_path in _normalize_paths(file_paths):
        with current_path.open("r", encoding=encoding) as f:
            iterator = _iter_with_progress(
                f,
                show_progress=show_progress,
                desc=f"{desc_prefix} ({current_path.name})",
            )
            for line in iterator:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _find_word_by_id(words: list[dict[str, Any]], token_id: int) -> dict[str, Any] | None:
    for word in words:
        if isinstance(word.get("id"), int) and word["id"] == token_id:
            return word
    return None


def _get_ordered_subtree_ids(
    words_by_id: dict[int, dict[str, Any]],
    head_id: int,
) -> list[int]:
    collected = []
    stack = [head_id]

    while stack:
        current = stack.pop()
        collected.append(current)
        children = [
            word["id"]
            for word in words_by_id.values()
            if word.get("head") == current and isinstance(word.get("id"), int)
        ]
        stack.extend(children)

    return sorted(set(collected))


def _collect_subtree_text(
    words_by_id: dict[int, dict[str, Any]],
    head_id: int,
) -> str:
    ordered_ids = _get_ordered_subtree_ids(words_by_id, head_id)
    return " ".join(
        words_by_id[idx]["text"] for idx in ordered_ids if idx in words_by_id
    )


def _join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens).strip()


def _filter_nested_clause_heads(
    words_by_id: dict[int, dict[str, Any]],
    clause_heads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    subtree_by_head_id = {
        clause_head["id"]: set(_get_ordered_subtree_ids(words_by_id, clause_head["id"]))
        for clause_head in clause_heads
    }

    filtered_heads = []
    for clause_head in clause_heads:
        is_nested = any(
            clause_head["id"] in subtree_ids and clause_head["id"] != other_head["id"]
            for other_head, subtree_ids in (
                (candidate, subtree_by_head_id[candidate["id"]]) for candidate in clause_heads
            )
        )
        if not is_nested:
            filtered_heads.append(clause_head)

    return filtered_heads


def compute_verbs_before_bahwa_frequencies_from_jsonl(
    parsed_jsonl_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    counter = Counter()
    total_matches = 0

    sentence_iter = _iter_jsonl_records(
        parsed_jsonl_path,
        encoding=encoding,
        show_progress=show_progress,
        desc_prefix="Counting verbs before bahwa",
    )

    for record in sentence_iter:
        words = record.get("words", [])
        words_by_id = {
            word["id"]: word for word in words if isinstance(word.get("id"), int)
        }

        for word in words:
            if not isinstance(word.get("id"), int):
                continue
            if word.get("upos") not in {"VERB", "AUX"}:
                continue

            next_word = words_by_id.get(word["id"] + 1)
            if next_word is None or (next_word.get("text") or "").lower() != "bahwa":
                continue

            surface_form = (word.get("text") or "").lower().strip()
            if not surface_form:
                continue

            counter[surface_form] += 1
            total_matches += 1

    rows = []
    for surface_form, count in counter.items():
        relative_frequency = count / total_matches if total_matches else 0.0
        rows.append(
            {
                "verb_surface": surface_form,
                "verb_before_bahwa_count": count,
                "verb_before_bahwa_relative_frequency": relative_frequency,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "verb_surface",
                "verb_before_bahwa_count",
                "verb_before_bahwa_relative_frequency",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        "verb_before_bahwa_count",
        ascending=False,
    ).reset_index(drop=True)


def compute_complement_taking_rates_from_jsonl(
    parsed_jsonl_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    target_surfaces: list[str] | tuple[str, ...] | set[str],
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    normalized_targets = {
        str(surface).lower().strip() for surface in target_surfaces if str(surface).strip()
    }
    columns = [
        "verb_surface",
        "target_verb_total_count",
        "complement_taking_count",
        "complement_taking_ratio",
        "complement_taking_ratio_z",
    ]
    if not normalized_targets:
        return pd.DataFrame(columns=columns)

    target_relations = {"ccomp", "xcomp", "parataxis", "conj", "advcl"}
    total_counter = Counter({surface: 0 for surface in normalized_targets})
    complement_counter = Counter({surface: 0 for surface in normalized_targets})

    sentence_iter = _iter_jsonl_records(
        parsed_jsonl_path,
        encoding=encoding,
        show_progress=show_progress,
        desc_prefix="Counting complement-taking rates",
    )

    for record in sentence_iter:
        words = record.get("words", [])
        words_by_id = {
            word["id"]: word for word in words if isinstance(word.get("id"), int)
        }

        for word in words:
            word_id = word.get("id")
            if not isinstance(word_id, int):
                continue

            surface_form = (word.get("text") or "").lower().strip()
            if surface_form not in normalized_targets:
                continue

            total_counter[surface_form] += 1
            clause_heads = [
                child
                for child in words
                if isinstance(child.get("id"), int)
                and child.get("head") == word_id
                and child.get("deprel") in target_relations
                and child["id"] > word_id
            ]
            clause_heads = _filter_nested_clause_heads(words_by_id, clause_heads)
            if clause_heads:
                complement_counter[surface_form] += 1

    rows = []
    for surface_form in sorted(normalized_targets):
        total_count = total_counter[surface_form]
        complement_count = complement_counter[surface_form]
        ratio = complement_count / total_count if total_count else math.nan
        rows.append(
            {
                "verb_surface": surface_form,
                "target_verb_total_count": total_count,
                "complement_taking_count": complement_count,
                "complement_taking_ratio": ratio,
            }
        )

    result = pd.DataFrame(rows, columns=columns[:-1])
    valid = result["complement_taking_ratio"].dropna()
    if valid.empty:
        result["complement_taking_ratio_z"] = pd.NA
        return result

    mean = valid.mean()
    std = valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        result["complement_taking_ratio_z"] = result["complement_taking_ratio"].where(
            result["complement_taking_ratio"].isna(),
            0.0,
        )
    else:
        result["complement_taking_ratio_z"] = (
            result["complement_taking_ratio"] - mean
        ) / std

    return result


def _extract_rows_for_sentence(
    record: dict[str, Any],
    target_verb_info: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    words = record.get("words", [])
    words_by_id = {
        word["id"]: word for word in words if isinstance(word.get("id"), int)
    }
    sentence_text = record.get("sentence") or _join_tokens(
        [token.get("text", "") for token in record.get("tokens", [])]
    )

    target_relations = {"ccomp", "xcomp", "parataxis", "conj", "advcl"}
    subject_relations = {"nsubj", "csubj", "nsubj:pass"}

    for word in words:
        word_id = word.get("id")
        if not isinstance(word_id, int):
            continue

        form = (word.get("text") or "").lower().strip()
        if form not in target_verb_info:
            continue

        verb_info = target_verb_info[form]
        next_word = _find_word_by_id(words, word_id + 1)
        marker_type = "No"
        has_bahwa_after_trigger = bool(
            next_word and (next_word.get("text") or "").lower() == "bahwa"
        )
        has_comma_after_trigger = bool(next_word and next_word.get("text") == ",")

        if has_bahwa_after_trigger:
            marker_type = "bahwa"
        elif has_comma_after_trigger:
            marker_type = "comma"

        clause_heads = [
            child
            for child in words
            if isinstance(child.get("id"), int)
            and child.get("head") == word_id
            and child.get("deprel") in target_relations
            and child["id"] > word_id
        ]
        clause_heads = _filter_nested_clause_heads(words_by_id, clause_heads)

        for clause_head in clause_heads:
            clause_head_id = clause_head["id"]
            subtree_ids = _get_ordered_subtree_ids(words_by_id, clause_head_id)
            clause_tokens = [
                words_by_id[idx]["text"] for idx in subtree_ids if idx in words_by_id
            ]
            clause_text = _join_tokens(clause_tokens)
            has_bahwa_in_clause = "bahwa" in [
                token.lower() for token in clause_tokens
            ]
            has_bahwa = int(has_bahwa_after_trigger or has_bahwa_in_clause)

            excluded_onset_ids = set()
            if has_bahwa_after_trigger or has_comma_after_trigger:
                excluded_onset_ids.add(word_id + 1)

            sc_onset_id = next(
                (
                    idx
                    for idx in subtree_ids
                    if idx in words_by_id
                    and idx not in excluded_onset_ids
                    and (words_by_id[idx].get("text") or "").lower() != "bahwa"
                    and words_by_id[idx].get("text") != ","
                ),
                None,
            )
            if sc_onset_id is None:
                continue

            sc_onset = words_by_id[sc_onset_id]["text"]
            mc_tokens = [
                words_by_id[idx]["text"] for idx in sorted(words_by_id) if idx < sc_onset_id
            ]
            if mc_tokens and mc_tokens[-1].lower() == "bahwa":
                mc_tokens = mc_tokens[:-1]
            if mc_tokens and mc_tokens[-1] == ",":
                mc_tokens = mc_tokens[:-1]

            main_clause_text = _join_tokens(mc_tokens)
            words_before_trigger = len(
                [idx for idx in sorted(words_by_id) if idx < word_id]
            )
            subordinate_clause_word_count = len(clause_tokens)

            subjects = [
                child
                for child in words
                if isinstance(child.get("id"), int)
                and child.get("head") == clause_head_id
                and child.get("deprel") in subject_relations
            ]
            if not subjects:
                continue

            subject = min(subjects, key=lambda child: child["id"])
            subject_text = _collect_subtree_text(words_by_id, subject["id"])

            rows.append(
                {
                    "source_file": record.get("source_file"),
                    "sentence_id": record.get("sentence_id"),
                    "sentence": sentence_text,
                    "trigger": word.get("text"),
                    "target_surface": form,
                    "trigger_lemma": (word.get("lemma") or "").lower().strip(),
                    "clause_relation": clause_head.get("deprel"),
                    "marker_type": marker_type,
                    "has_bahwa": has_bahwa,
                    "main_clause_text": main_clause_text,
                    "words_before_trigger": words_before_trigger,
                    "subordinate_clause_word_count": subordinate_clause_word_count,
                    "sc_onset": sc_onset,
                    "subject": subject_text,
                    "predicate": clause_head.get("text"),
                    "clause_text": clause_text,
                    "trigger_id": word_id,
                    "clause_head_id": clause_head_id,
                    "subject_id": subject["id"],
                    "sc_onset_id": sc_onset_id,
                    "subject_deprel": subject.get("deprel"),
                    "verb_rank": verb_info["verb_rank"],
                    "verb_before_bahwa_count": verb_info["verb_before_bahwa_count"],
                    "verb_before_bahwa_relative_frequency": verb_info[
                        "verb_before_bahwa_relative_frequency"
                    ],
                }
            )

    return rows


def build_top_verb_parse_dataset_from_jsonl(
    parsed_jsonl_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    output_csv_path: str | Path,
    top_n: int = 10,
    start_rank: int = 1,
    end_rank: int | None = None,
    reference_parsed_jsonl_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    if reference_parsed_jsonl_path is None:
        reference_parsed_jsonl_path = parsed_jsonl_path

    verb_freq_df = compute_verbs_before_bahwa_frequencies_from_jsonl(
        reference_parsed_jsonl_path,
        encoding=encoding,
        show_progress=show_progress,
    ).reset_index(drop=True)
    verb_freq_df["verb_rank"] = range(1, len(verb_freq_df) + 1)

    if start_rank < 1:
        raise ValueError("start_rank must be 1 or greater.")

    if end_rank is not None and end_rank < start_rank:
        raise ValueError("end_rank must be greater than or equal to start_rank.")

    if end_rank is None:
        end_rank = start_rank + top_n - 1

    top_verbs_df = verb_freq_df[
        (verb_freq_df["verb_rank"] >= start_rank)
        & (verb_freq_df["verb_rank"] <= end_rank)
    ].copy()
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if top_verbs_df.empty:
        empty_df = pd.DataFrame()
        empty_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved parsed CSV to: {output_csv_path}")
        return empty_df

    target_verb_info = {
        str(row.verb_surface).lower(): {
            "verb_rank": row.verb_rank,
            "verb_before_bahwa_count": row.verb_before_bahwa_count,
            "verb_before_bahwa_relative_frequency": row.verb_before_bahwa_relative_frequency,
        }
        for row in top_verbs_df.itertuples(index=False)
    }

    collected_rows = []
    sentence_iter = _iter_jsonl_records(
        parsed_jsonl_path,
        encoding=encoding,
        show_progress=show_progress,
        desc_prefix="Extracting clauses from parsed JSONL",
    )

    for record in sentence_iter:
        collected_rows.extend(_extract_rows_for_sentence(record, target_verb_info))

    if collected_rows:
        combined_df = pd.DataFrame(collected_rows)
        combined_df = combined_df.sort_values(
            by=["verb_rank", "source_file", "sentence_id", "trigger_id", "clause_head_id"]
        ).reset_index(drop=True)
    else:
        combined_df = pd.DataFrame()

    combined_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved parsed CSV to: {output_csv_path}")
    return combined_df


if __name__ == "__main__":
    parsed_jsonl_path = "/content/drive/MyDrive/300K_ind_parsed.jsonl"
    output_csv_path = "/content/LSJ_0618.csv"

    final_df = build_top_verb_parse_dataset_from_jsonl(
        parsed_jsonl_path=parsed_jsonl_path,
        output_csv_path=output_csv_path,
        start_rank=1,
        end_rank=30,
    )
    print(final_df.head().to_string(index=False))

if not final_df.empty:
        verb_count_df = (
            final_df["target_surface"]
            .value_counts()
            .rename_axis("verb_surface")
            .reset_index(name="extracted_count")
        )
        print("\n動詞ごとの抽出数（多い順）")
        print(verb_count_df.to_string(index=False))
else:
        print("\n抽出結果が空のため、動詞ごとの件数はありません。")
