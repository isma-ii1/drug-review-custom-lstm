"""Predict a rating-derived sentiment from review text only."""
import argparse
import json
from pathlib import Path

import torch

from .data import LABELS, tokenize, file_hash
from .models import ReviewClassifier


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--vocabulary",default="data/processed/vocabulary.json")
    parser.add_argument("--max-length",type=int,default=160)
    parser.add_argument("--text",required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    c=checkpoint["config"]
    vocab=json.loads(Path(args.vocabulary).read_text(encoding="utf-8"))
    if len(vocab)!=checkpoint["vocab_size"]:
        raise ValueError("Vocabulary does not match the checkpoint")
    signature_path=Path(args.checkpoint).resolve().parent.parent/"data_signature.json"
    if signature_path.exists():
        signature=json.loads(signature_path.read_text())
        if file_hash(args.vocabulary)!=signature["vocabulary_sha256"]:
            raise ValueError("Vocabulary content does not match the trained experiment")
    tokens=tokenize(args.text)[:args.max_length]
    if not tokens:
        parser.error("Review must contain at least one word or number")
    model=ReviewClassifier(len(vocab),c["embedding_dim"],c["hidden_size"],c["variant"],c["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x=torch.tensor([[vocab.get(t,1) for t in tokens]])
    with torch.no_grad():
        probabilities=model(x,torch.tensor([len(tokens)])).softmax(-1)[0].tolist()
    print(json.dumps({"predicted_label":LABELS[max(range(3),key=lambda i:probabilities[i])],
        "softmax_scores":dict(zip(LABELS,probabilities)),
        "note":"Rating-derived review classification. Scores are uncalibrated."},indent=2))


if __name__=="__main__":
    main()
