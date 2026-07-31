from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import stanza
except ImportError:
    stanza = None

try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None


def build_id_pipeline():
    """Create an Indonesian Stanza pipeline for tokenization, POS, lemmatization, and dependencies."""
    if stanza is None:
        raise ImportError(
            "stanza is not installed. In Colab, run: !pip install stanza"
        )

    use_gpu = bool(torch is not None and torch.cuda.is_available())
    print(f"Using {'GPU' if use_gpu else 'CPU'} for Stanza pipeline.")

    return stanza.Pipeline(
        lang="id",
        processors="tokenize,pos,lemma,depparse",
        tokenize_no_ssplit=False,
        use_gpu=use_gpu,
    )


def _word_attr(word: Any, key: str, default=None):
    if isinstance(word, dict):
        return word.get(key, default)
    return getattr(word, key, default)


def _collect_subtree_text(words_by_id: dict[int, Any], head_id: int) -> str:
    collected = []
    stack = [head_id]

    while stack:
        current = stack.pop()
        collected.append(current)
        children = [
            _word_attr(word, "id")
            for word in words_by_id.values()
            if _word_attr(word, "head") == current
            and isinstance(_word_attr(word, "id"), int)
        ]
        stack.extend(children)

    ordered_ids = sorted(set(collected))
    return " ".join(
        _word_attr(words_by_id[idx], "text", "")
        for idx in ordered_ids
        if idx in words_by_id
    )


def _find_token_by_id(words: list[Any], token_id: int):
    for word in words:
        word_id = _word_attr(word, "id")
        if isinstance(word_id, int) and word_id == token_id:
            return word
    return None


def _get_ordered_subtree_ids(words_by_id: dict[int, Any], head_id: int) -> list[int]:
    collected = []
    stack = [head_id]

    while stack:
        current = stack.pop()
        collected.append(current)
        children = [
            _word_attr(word, "id")
            for word in words_by_id.values()
            if _word_attr(word, "head") == current
            and isinstance(_word_attr(word, "id"), int)
        ]
        stack.extend(children)

    return sorted(set(collected))


def _join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens).strip()


def _iter_with_progress(items, show_progress: bool, desc: str):
    if not show_progress:
        return items

    if tqdm is not None:
        return tqdm(items, desc=desc)

    total = len(items)

    def generator():
        for idx, item in enumerate(items, start=1):
            if idx == 1 or idx % 50 == 0 or idx == total:
                print(f"{desc}: {idx}/{total}")
            yield item

    return generator()


def build_lm(
    model_name: str = "cahya/gpt2-large-indonesian-522M",
):
    """Load the autoregressive LM used for surprisal and entropy calculations."""
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is not installed. In Colab, run: !pip install transformers"
        )
    if torch is None:
        raise ImportError(
            "torch is not installed. In Colab, run: !pip install torch"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LM on {device}: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    model.to(device)
    model.eval()

    return tokenizer, model, device


def _get_last_token_logits(prefix_text: str, tokenizer, model, device):
    encoded = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    return outputs.logits[0, -1, :]


def _word_subtoken_ids(word_text: str, tokenizer) -> list[int]:
    token_ids = tokenizer.encode(" " + word_text, add_special_tokens=False)
    if not token_ids:
        token_ids = tokenizer.encode(word_text, add_special_tokens=False)
    return token_ids


def _subword_surprisal_and_entropy(
    prefix_text: str,
    word_text: str,
    tokenizer,
    model,
    device,
):
    subtoken_ids = _word_subtoken_ids(word_text, tokenizer)
    if not subtoken_ids:
        return math.nan, math.nan

    running_prefix = prefix_text
    surprisal = 0.0
    entropy = 0.0

    for subtoken_id in subtoken_ids:
        logits = _get_last_token_logits(running_prefix, tokenizer, model, device)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        surprisal += float(-log_probs[subtoken_id].item())
        entropy += float(-(probs * log_probs).sum().item())

        subtoken_text = tokenizer.decode([subtoken_id], clean_up_tokenization_spaces=False)
        running_prefix += subtoken_text

    return surprisal, entropy


def _sc_onset_prefix_variants(mc_text: str) -> list[str]:
    common_prefix = str(mc_text).strip()
    return [
        common_prefix,
        f"{common_prefix} bahwa".strip(),
        f"{common_prefix} ,".strip(),
    ]


def _marginalized_sc_onset_metrics(
    mc_text: str,
    sc_onset: str,
    tokenizer,
    model,
    device,
) -> tuple[float, float]:
    total_prob = 0.0
    total_entropy = 0.0
    valid_count = 0

    for prefix_variant in _sc_onset_prefix_variants(mc_text):
        surprisal, entropy = _subword_surprisal_and_entropy(
            prefix_variant,
            sc_onset,
            tokenizer,
            model,
            device,
        )
        if math.isnan(surprisal) or math.isnan(entropy):
            continue

        total_prob += math.exp(-surprisal)
        total_entropy += entropy
        valid_count += 1

    if total_prob <= 0.0 or valid_count == 0:
        return math.inf, math.nan

    return float(-math.log(total_prob / valid_count)), total_entropy / valid_count


def _marginalized_sc_onset_surprisal(
    mc_text: str,
    sc_onset: str,
    tokenizer,
    model,
    device,
) -> float:
    surprisal, _ = _marginalized_sc_onset_metrics(
        mc_text,
        sc_onset,
        tokenizer,
        model,
        device,
    )
    return surprisal


def add_uid_metrics(
    df: pd.DataFrame,
    tokenizer=None,
    model=None,
    device: str | None = None,
    model_name: str = "cahya/gpt2-large-indonesian-522M",
    show_progress: bool = True,
) -> pd.DataFrame:
    if df.empty:
        result = df.copy()
        result["sc_onset_surprisal"] = pd.Series(dtype=float)
        result["sc_onset_entropy"] = pd.Series(dtype=float)
        return result

    if tokenizer is None or model is None or device is None:
        tokenizer, model, device = build_lm(model_name=model_name)

    result = df.copy()
    surprisals = []
    entropies = []
    row_iter = _iter_with_progress(
        list(result.itertuples(index=False)),
        show_progress=show_progress,
        desc="Computing marginalized UID metrics",
    )

    for row in row_iter:
        mc_text = getattr(row, "main_clause_text")
        sc_onset = getattr(row, "sc_onset")

        surprisal, entropy = _marginalized_sc_onset_metrics(
            mc_text,
            sc_onset,
            tokenizer,
            model,
            device,
        )

        surprisals.append(surprisal)
        entropies.append(entropy)

    result["sc_onset_surprisal"] = surprisals
    result["sc_onset_entropy"] = entropies
    return result


def _get_sentence_main_verb_lemma(sentence) -> str | None:
    verbal_roots = [
        word
        for word in sentence.words
        if word.head == 0 and word.upos in {"VERB", "AUX"}
    ]
    if not verbal_roots:
        return None

    root = verbal_roots[0]
    lemma = (root.lemma or root.text or "").lower().strip()
    return lemma or None


def compute_main_verb_frequencies(
    reference_text_path: str,
    nlp=None,
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    if nlp is None:
        nlp = build_id_pipeline()

    with open(reference_text_path, "r", encoding=encoding) as f:
        text = f.read()

    if show_progress:
        print("Parsing reference corpus for main verb frequencies...")
    doc = nlp(text)
    if show_progress:
        print(f"Finished parsing. Sentences to inspect: {len(doc.sentences)}")

    counter = Counter()
    total_sentences_with_main_verb = 0

    sentence_iter = _iter_with_progress(
        doc.sentences,
        show_progress=show_progress,
        desc="Counting main verbs",
    )

    for sentence in sentence_iter:
        lemma = _get_sentence_main_verb_lemma(sentence)
        if lemma is None:
            continue
        counter[lemma] += 1
        total_sentences_with_main_verb += 1

    rows = []
    for lemma, count in counter.items():
        relative_frequency = count / total_sentences_with_main_verb
        rows.append(
            {
                "verb_lemma": lemma,
                "main_verb_count": count,
                "main_verb_relative_frequency": relative_frequency,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "main_verb_count",
        ascending=False,
    ).reset_index(drop=True)


def standardize_numeric_columns(
    input_csv_path: str,
    output_csv_path: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)
    result = df.copy()

    excluded_columns = {"sentence_id", "has_bahwa", "has_kalau"}
    numeric_columns = [
        column
        for column in result.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]

    for column in numeric_columns:
        std = result[column].std(ddof=0)
        mean = result[column].mean()

        if pd.isna(std) or std == 0:
            result[column] = 0.0
        else:
            result[column] = (result[column] - mean) / std

    if output_csv_path is None:
        input_path = Path(input_csv_path)
        output_csv_path = str(
            input_path.with_name(f"{input_path.stem}_standardized{input_path.suffix}")
        )

    result.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved standardized CSV to: {output_csv_path}")

    return result


def compute_verbs_before_bahwa_frequencies(
    file_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    nlp=None,
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Count verb surface forms immediately followed by 'bahwa'.
    If multiple files are provided, frequencies are computed over the merged corpus.
    """
    if nlp is None:
        nlp = build_id_pipeline()

    counter = Counter()
    total_matches = 0
    file_paths = (
        [file_path]
        if isinstance(file_path, (str, Path))
        else list(file_path)
    )

    for current_path in file_paths:
        with open(current_path, "r", encoding=encoding) as f:
            text = f.read()

        if show_progress:
            print(f"Parsing text for verbs immediately before 'bahwa': {current_path}")
        doc = nlp(text)
        if show_progress:
            print(f"Finished parsing. Sentences to inspect: {len(doc.sentences)}")

        sentence_iter = _iter_with_progress(
            doc.sentences,
            show_progress=show_progress,
            desc=f"Counting verbs before bahwa ({Path(current_path).name})",
        )

        for sentence in sentence_iter:
            words = sentence.words
            words_by_id = {word.id: word for word in words if isinstance(word.id, int)}

            for word in words:
                if not isinstance(word.id, int):
                    continue
                if word.upos not in {"VERB", "AUX"}:
                    continue

                next_word = words_by_id.get(word.id + 1)
                if next_word is None or next_word.text.lower() != "bahwa":
                    continue

                surface_form = (word.text or "").lower().strip()
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
