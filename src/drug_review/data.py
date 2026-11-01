"""Deterministic cleaning, deduplication and train-only vocabulary fitting."""
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

LABELS = ["Negative", "Neutral", "Positive"]
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[!?]")


def normalize(text):
    text = html.unescape(str(text)).lower().replace("\u2019", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip().strip('\"').strip()


def tokenize(text):
    return TOKEN.findall(normalize(text))


def rating_label(rating):
    rating = float(rating)
    if not rating.is_integer() or not 1 <= rating <= 10:
        raise ValueError("Ratings must be integers from 1 to 10")
    return 0 if rating <= 4 else (1 if rating <= 6 else 2)


def read_source(path):
    frame = pd.read_csv(path)
    required = {"uniqueID", "review", "rating"}
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {required - set(frame)}")
    rating = pd.to_numeric(frame["rating"], errors="coerce")
    valid = frame.review.notna() & rating.between(1, 10) & rating.eq(rating.round())
    frame["normalized"] = frame.review.fillna("").map(normalize)
    valid &= frame.normalized.map(lambda s: bool(TOKEN.search(s)))
    quality = {"raw_rows": len(frame), "columns": list(frame.columns[:-1]),
               "missing": {k: int(v) for k, v in frame.isna().sum().items()},
               "duplicate_ids": int(frame.uniqueID.duplicated().sum()),
               "duplicate_text_rows": int(frame.normalized.duplicated().sum()),
               "invalid_required_rows": int((~valid).sum()),
               "rating_counts": {str(k): int(v) for k,v in rating.value_counts().sort_index().items()}}
    frame = frame.loc[valid].copy()
    frame["label"] = frame.rating.map(rating_label)
    conflicts = frame.groupby("normalized").label.nunique()
    conflict_texts = set(conflicts[conflicts > 1].index)
    quality["conflicting_texts"] = len(conflict_texts)
    quality["conflicting_rows_removed"] = int(frame.normalized.isin(conflict_texts).sum())
    frame = frame.loc[~frame.normalized.isin(conflict_texts)].copy()
    before = len(frame)
    frame = frame.drop_duplicates("normalized").reset_index(drop=True)
    quality["same_label_duplicates_removed"] = before - len(frame)
    quality["clean_unique_rows"] = len(frame)
    return frame, quality


def subset(frame, limit, seed):
    if limit <= 0 or limit >= len(frame):
        return frame.reset_index(drop=True)
    selected, _ = train_test_split(frame, train_size=limit, stratify=frame.label, random_state=seed)
    return selected.reset_index(drop=True)


def build_data(train_path, test_path, out_dir, train_limit=15000, val_limit=3000,
               vocab_size=12000, max_length=160, seed=42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, train_quality = read_source(train_path)
    test, test_quality = read_source(test_path)
    # Exclude ALL raw test texts from training, including ambiguous test groups.
    raw_test = pd.read_csv(test_path, usecols=["review"]).review.dropna().map(normalize)
    overlap = train.normalized.isin(set(raw_test))
    train_quality["rows_overlapping_raw_test_removed"] = int(overlap.sum())
    train = train.loc[~overlap].reset_index(drop=True)
    train_pool, val_pool = train_test_split(train, test_size=0.2, stratify=train.label, random_state=seed)
    train = subset(train_pool, train_limit, seed)
    val = subset(val_pool, val_limit, seed + 1)
    splits = {"train": train, "validation": val, "test": test}
    tokenized = {name: [tokenize(s) for s in frame.normalized] for name, frame in splits.items()}
    counts = Counter(t for row in tokenized["train"] for t in row)
    ordered = sorted(counts, key=lambda x: (-counts[x], x))[:vocab_size - 2]
    vocab = {"<PAD>": 0, "<UNK>": 1, **{word: i + 2 for i, word in enumerate(ordered)}}
    arrays = {}
    audit = {"train_source": train_quality, "test_source": test_quality, "seed": seed,
             "train_pool_rows": len(train_pool), "validation_pool_rows": len(val_pool),
             "max_length": max_length, "vocab_size": len(vocab), "splits": {},
             "source_sha256": {"train": file_hash(train_path), "test": file_hash(test_path)}}
    manifests = []
    for name, frame in splits.items():
        toks = tokenized[name]
        lengths = np.array([min(len(t), max_length) for t in toks], dtype=np.int64)
        x = np.zeros((len(toks), max_length), dtype=np.int64)
        for i, tokens in enumerate(toks):
            x[i, :lengths[i]] = [vocab.get(t, 1) for t in tokens[:max_length]]
        arrays[name + "_tokens"] = x
        arrays[name + "_lengths"] = lengths
        arrays[name + "_labels"] = frame.label.to_numpy(dtype=np.int64)
        raw_lengths = np.array([len(t) for t in toks])
        audit["splits"][name] = {"rows": len(frame),
            "class_counts": {LABELS[i]: int((frame.label == i).sum()) for i in range(3)},
            "token_length_quantiles": dict(zip(["p25", "p50", "p75", "p90", "p95", "p99"],
                                              np.quantile(raw_lengths, [.25,.5,.75,.9,.95,.99]).tolist())),
            "truncation_rate": float((raw_lengths > max_length).mean()),
            "unknown_token_rate": float(((x == 1).sum()) / lengths.sum())}
        manifests.append(pd.DataFrame({"split": name, "uniqueID": frame.uniqueID,
            "text_sha256": frame.normalized.map(lambda s: hashlib.sha256(s.encode()).hexdigest()),
            "label": frame.label}))
        # Error analysis needs only test text. Do not use metadata as input.
        if name == "test":
            frame[["uniqueID", "review", "label"]].to_csv(out / "test_records.csv", index=False)
    manifest = pd.concat(manifests, ignore_index=True)
    assert not manifest.text_sha256.duplicated().any(), "Cross-split text leakage"
    manifest.to_csv(out / "split_manifest.csv", index=False)
    np.savez_compressed(out / "encoded.npz", **arrays)
    (out / "vocabulary.json").write_text(json.dumps(vocab), encoding="utf-8")
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    np.save(out / "training_lengths.npy", np.array([len(t) for t in tokenized["train"]]))
    return audit


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_loader(arrays, name, batch_size, shuffle=False, seed=42):
    data = torch.utils.data.TensorDataset(*[
        torch.from_numpy(arrays[name + suffix].copy()) for suffix in ["_tokens", "_lengths", "_labels"]])
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=0, collate_fn=collate)


def collate(rows):
    x, lengths, y = [torch.stack(group) for group in zip(*rows)]
    return x[:, :int(lengths.max())], lengths, y
