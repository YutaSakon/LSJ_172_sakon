from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def build_id_pipeline(download_model: bool = True):
    """
    Create an Indonesian Stanza pipeline that is ready to run in Colab.

    When `download_model` is True, missing Stanza resources are downloaded
    automatically before the pipeline is created.
    """
    if stanza is None:
        raise ImportError(
            "stanza is not installed. In Colab, run: !pip install stanza"
        )

    if download_model:
        stanza.download("id")

    use_gpu = bool(torch is not None and torch.cuda.is_available())
    print(f"Using {'GPU' if use_gpu else 'CPU'} for Stanza pipeline.")

    return stanza.Pipeline(
        lang="id",
        processors="tokenize,pos,lemma,depparse",
        tokenize_no_ssplit=False,
        use_gpu=use_gpu,
    )


def _serialize_token_id(token_id: Any) -> int | list[int] | None:
    if token_id is None:
        return None
    if isinstance(token_id, int):
        return token_id
    if isinstance(token_id, tuple):
        return list(token_id)
    if isinstance(token_id, list):
        return token_id
    return list(token_id)


def _serialize_word(word: Any) -> dict[str, Any]:
    return {
        "id": word.id,
        "text": word.text,
        "lemma": word.lemma,
        "upos": word.upos,
        "head": word.head,
        "deprel": word.deprel,
    }


def _serialize_token(token: Any) -> dict[str, Any]:
    return {
        "id": _serialize_token_id(token.id),
        "text": token.text,
    }


def _serialize_sentence(sentence: Any, sentence_id: int, source_file: str) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "sentence_id": sentence_id,
        "sentence": " ".join(token.text for token in sentence.tokens),
        "tokens": [_serialize_token(token) for token in sentence.tokens],
        "words": [_serialize_word(word) for word in sentence.words],
    }


def save_parsed_corpus(
    input_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    output_path: str | Path,
    nlp=None,
    encoding: str = "utf-8",
    show_progress: bool = True,
) -> Path:
    """
    Parse one or more text files with Stanza and save sentence-level analyses as JSONL.

    Each line in the output file contains one sentence with token- and word-level
    fields needed for downstream clause extraction.
    """
    if nlp is None:
        nlp = build_id_pipeline()

    input_paths = [input_path] if isinstance(input_path, (str, Path)) else list(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_sentences = 0
    with output_file.open("w", encoding="utf-8") as out_f:
        for current_path in input_paths:
            current_path = Path(current_path)
            with current_path.open("r", encoding=encoding) as in_f:
                text = in_f.read()

            if show_progress:
                print(f"Parsing full corpus: {current_path}")
            doc = nlp(text)
            if show_progress:
                print(f"Finished parsing. Sentences to save: {len(doc.sentences)}")

            sentence_iter = _iter_with_progress(
                list(enumerate(doc.sentences, start=1)),
                show_progress=show_progress,
                desc=f"Saving parsed sentences ({current_path.name})",
            )

            for sentence_id, sentence in sentence_iter:
                record = _serialize_sentence(
                    sentence=sentence,
                    sentence_id=sentence_id,
                    source_file=str(current_path),
                )
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_sentences += 1

    if show_progress:
        print(f"Saved {total_sentences} parsed sentences to: {output_file}")

    return output_file


if __name__ == "__main__":
    input_path = [
        "/content/ind-id_web_2017_300K-sentences.txt",
        "/content/ind_news_2019_300K-sentences.txt",
    ]
    output_path = "/content/drive/MyDrive/ind_parsed.jsonl"

    save_parsed_corpus(
        input_path=input_path,
        output_path=output_path,
    )
