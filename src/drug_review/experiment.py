"""Reproducible experiments. Validation selection precedes all test scoring."""
import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

from .data import LABELS, build_data, make_loader, file_hash
from .models import ReviewClassifier, parameter_counts


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    truth, pred, losses = [], [], []
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        losses.append((float(loss_fn(logits, y)), len(y)))
        truth.extend(y.cpu().tolist())
        pred.extend(logits.argmax(-1).cpu().tolist())
    return {"accuracy": accuracy_score(truth, pred),
            "macro_f1": f1_score(truth, pred, labels=[0,1,2], average="macro", zero_division=0),
            "loss": sum(v*n for v,n in losses)/sum(n for _,n in losses)}, truth, pred


def default_configs(epochs=6, batch_size=256):
    # An equal 4-configuration search for every architecture. Each ablation
    # changes just one factor from fixed to make effects interpretable.
    base = {"learning_rate": .003, "batch_size": batch_size, "optimizer": "adamw",
            "dropout": .2, "weight_decay": .01, "epochs": epochs,
            "embedding_dim": 64, "hidden_size": 32}
    settings = [dict(base, setting="fixed"),
                dict(base, setting="lr_0.001", learning_rate=.001),
                dict(base, setting="batch_128", batch_size=128),
                dict(base, setting="adam", optimizer="adam")]
    return [dict(c, variant=v, run_id=f"{v}_{c['setting']}")
            for v in ["vanilla", "cifg", "wmc"] for c in settings]


def run_training(config, arrays, vocab_size, out, device, seed, patience):
    seed_everything(seed)
    run_dir = out / "runs" / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoints" / (config["run_id"] + ".pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = ReviewClassifier(vocab_size, config["embedding_dim"], config["hidden_size"],
                             config["variant"], config["dropout"]).to(device)
    counts = parameter_counts(model)
    # Script the manual loop for lower Python overhead. Remains custom nn.Linear.
    compiled = torch.jit.script(model)
    optimizer_class = torch.optim.AdamW if config["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(compiled.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    # Identical loss for all runs. Class weights are derived from training only.
    class_counts = np.bincount(arrays["train_labels"], minlength=3)
    weights = torch.tensor(len(arrays["train_labels"])/(3*class_counts), dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    # Report ordinary CE for comparable per-example evaluation loss.
    eval_loss = nn.CrossEntropyLoss()
    train_loader = make_loader(arrays,"train",config["batch_size"],True,seed)
    val_loader = make_loader(arrays,"validation",256)
    best, bad_epochs, history = -1., 0, []
    start = time.perf_counter()
    for epoch in range(1,config["epochs"]+1):
        compiled.train()
        epoch_start, running, n = time.perf_counter(), 0., 0
        for x,lengths,y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = compiled(x,lengths)
            loss = loss_fn(logits,y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {config['run_id']}")
            loss.backward()
            nn.utils.clip_grad_norm_(compiled.parameters(),1.)
            optimizer.step()
            running += float(loss.detach())*len(y)
            n += len(y)
        metrics,_,_ = evaluate(compiled,val_loader,device,eval_loss)
        row = {"epoch":epoch,"train_weighted_loss":running/n,"val_loss":metrics["loss"],
               "val_accuracy":metrics["accuracy"],"val_macro_f1":metrics["macro_f1"],
               "epoch_seconds":time.perf_counter()-epoch_start}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir/"history.csv",index=False)
        print(json.dumps({"run":config["run_id"],**row}),flush=True)
        if metrics["macro_f1"] > best + 1e-8:
            best, bad_epochs = metrics["macro_f1"], 0
            torch.save({"state_dict":model.state_dict(),"config":config,"epoch":epoch,
                        "validation":metrics,"vocab_size":vocab_size},checkpoint_path)
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    result = {**config,"seed":seed,"best_val_macro_f1":best,
              "best_epoch":max(history,key=lambda r:r["val_macro_f1"])["epoch"],
              "epochs_run":len(history),"training_seconds":time.perf_counter()-start,
              "total_parameters":counts["total"],"recurrent_parameters":counts["recurrent"],
              "embedding_parameters":counts["embedding"],"head_parameters":counts["head"],
              "checkpoint":str(checkpoint_path.relative_to(out))}
    json_write(run_dir/"result.json",result)
    return result


def evaluate_checkpoint(result, arrays, out, device, tag):
    checkpoint = torch.load(out/result["checkpoint"],map_location=device,weights_only=True)
    c = checkpoint["config"]
    model = ReviewClassifier(checkpoint["vocab_size"],c["embedding_dim"],c["hidden_size"],c["variant"],c["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model = torch.jit.script(model.to(device))
    start = time.perf_counter()
    metrics,truth,pred = evaluate(model,make_loader(arrays,"test",256),device,nn.CrossEntropyLoss())
    metrics["inference_seconds"] = time.perf_counter()-start
    metrics["classification_report"] = classification_report(truth,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(truth,pred,labels=[0,1,2]).tolist()
    metrics["run_id"] = result["run_id"]
    metrics["comparison"] = tag
    json_write(out/"evaluation"/(result["run_id"]+".json"),metrics)
    pd.DataFrame({"true_label":truth,"predicted_label":pred}).to_csv(out/"evaluation"/(result["run_id"]+"_predictions.csv"),index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--train",required=True)
    prep.add_argument("--test",required=True)
    prep.add_argument("--output",default="data/processed")
    prep.add_argument("--train-limit",type=int,default=15000,help="0 uses full eligible training pool")
    prep.add_argument("--val-limit",type=int,default=3000,help="0 uses full eligible validation pool")
    prep.add_argument("--max-length",type=int,default=160)
    prep.add_argument("--vocab-size",type=int,default=12000)
    prep.add_argument("--seed",type=int,default=42)
    run = sub.add_parser("train")
    run.add_argument("--data",default="data/processed")
    run.add_argument("--output",default="artifacts")
    run.add_argument("--epochs",type=int,default=6)
    run.add_argument("--patience",type=int,default=3)
    run.add_argument("--threads",type=int,default=2)
    run.add_argument("--seed",type=int,default=42)
    run.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    run.add_argument("--fixed-only",action="store_true")
    run.add_argument("--batch-size",type=int,default=256)
    args = parser.parse_args()
    if args.command == "prepare":
        audit=build_data(args.train,args.test,args.output,args.train_limit,args.val_limit,args.vocab_size,args.max_length,args.seed)
        print(json.dumps(audit,indent=2))
        return
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    (out/"evaluation").mkdir(parents=True,exist_ok=True)
    arrays = np.load(Path(args.data)/"encoded.npz")
    vocab = json.loads((Path(args.data)/"vocabulary.json").read_text())
    signature={"encoded_sha256":file_hash(Path(args.data)/"encoded.npz"),
               "vocabulary_sha256":file_hash(Path(args.data)/"vocabulary.json"),
               "seed":args.seed,"patience":args.patience}
    signature_path=out/"data_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text())!=signature:
        raise ValueError("Data or training policy changed. Choose a fresh output directory.")
    json_write(signature_path,signature)
    device = "cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    json_write(out/"environment.json",{"python":platform.python_version(),"torch":torch.__version__,
        "numpy":np.__version__,"pandas":pd.__version__,"platform":platform.platform(),
        "cpu":platform.processor(),"device":device,"torch_threads":args.threads,"seed":args.seed,
        "train_rows":len(arrays["train_labels"]),"validation_rows":len(arrays["validation_labels"]),
        "test_rows":len(arrays["test_labels"]),"cuda_available":torch.cuda.is_available()})
    configs=default_configs(args.epochs,args.batch_size)
    if args.fixed_only:
        configs=[c for c in configs if c["setting"]=="fixed"]
    json_write(out/"experiment_plan.json",configs)
    results=[]
    for config in configs:
        result_path=out/"runs"/config["run_id"]/"result.json"
        if result_path.exists():
            cached=json.loads(result_path.read_text())
            if all(cached.get(k)==v for k,v in config.items()) and cached.get("seed")==args.seed:
                results.append(cached)
                continue
            raise ValueError(f"Config changed for {config['run_id']}. Choose a new output directory.")
        result=run_training(config,arrays,len(vocab),out,device,args.seed,args.patience)
        results.append(result)
        pd.DataFrame(results).to_csv(out/"validation_results.csv",index=False)
    pd.DataFrame(results).to_csv(out/"validation_results.csv",index=False)
    # Freeze selection before viewing any test predictions.
    selections={variant:max([r for r in results if r["variant"]==variant],key=lambda r:r["best_val_macro_f1"])["run_id"]
                for variant in ["vanilla","cifg","wmc"]}
    json_write(out/"selection_before_test.json",selections)
    evaluations=[]
    for result in results:
        fixed=result["setting"]=="fixed"
        selected=result["run_id"]==selections[result["variant"]]
        if not (fixed or selected):
            continue
        metrics=evaluate_checkpoint(result,arrays,out,device,"fixed+selected" if fixed and selected else ("fixed" if fixed else "selected"))
        evaluations.append({**result,"test_accuracy":metrics["accuracy"],"test_macro_f1":metrics["macro_f1"],
                            "test_inference_seconds":metrics["inference_seconds"],"comparison":metrics["comparison"]})
    pd.DataFrame(evaluations).to_csv(out/"test_results.csv",index=False)
    majority=int(np.bincount(arrays["train_labels"]).argmax())
    truth=arrays["test_labels"]
    pred=np.full_like(truth,majority)
    json_write(out/"majority_baseline.json",{"class":LABELS[majority],"test_accuracy":accuracy_score(truth,pred),
        "test_macro_f1":f1_score(truth,pred,average="macro",labels=[0,1,2],zero_division=0)})
    print("COMPLETE: validation search and fixed/selected test comparisons saved",flush=True)


if __name__=="__main__":
    main()
