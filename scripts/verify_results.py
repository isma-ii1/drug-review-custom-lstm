"""Independent checks of persisted predictions and experiment selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parents[1]


def macro_f1_from_counts(cm):
    tp=np.diag(cm)
    denominator=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*tp,denominator,out=np.zeros(3),where=denominator!=0).mean())


def main():
    art=ROOT/"artifacts"
    validation=pd.read_csv(art/"validation_results.csv")
    test=pd.read_csv(art/"test_results.csv")
    selected=json.loads((art/"selection_before_test.json").read_text())
    selection_saved=(art/"selection_before_test.json").stat().st_mtime
    assert len(validation)==12 and validation.run_id.nunique()==12
    plan=json.loads((art/"experiment_plan.json").read_text())
    assert set(validation.run_id)=={c["run_id"] for c in plan}
    arrays=np.load(ROOT/"data/processed/encoded.npz")
    truth=arrays["test_labels"]
    manifest=pd.read_csv(ROOT/"data/processed/split_manifest.csv")
    assert not manifest.text_sha256.duplicated().any()
    for variant in ["vanilla","cifg","wmc"]:
        group=validation[validation.variant==variant]
        best=group.sort_values("best_val_macro_f1",ascending=False,kind="stable").iloc[0]
        assert selected[variant]==best.run_id
        params=group.total_parameters.unique()
        assert len(params)==1
    for row in test.itertuples():
        pred=pd.read_csv(art/"evaluation"/(row.run_id+"_predictions.csv"))
        assert (art/"evaluation"/(row.run_id+"_predictions.csv")).stat().st_mtime>=selection_saved
        assert len(pred)==len(truth)
        np.testing.assert_array_equal(pred.true_label,truth)
        yhat=pred.predicted_label.to_numpy()
        # Arithmetic derived from the count matrix, independent of sklearn F1.
        cm=np.zeros((3,3),dtype=np.int64)
        np.add.at(cm,(truth,yhat),1)
        accuracy=float(np.trace(cm)/cm.sum())
        macro=macro_f1_from_counts(cm)
        assert abs(accuracy-row.test_accuracy)<1e-12
        assert abs(macro-row.test_macro_f1)<1e-12
        metrics=json.loads((art/"evaluation"/(row.run_id+".json")).read_text())
        np.testing.assert_array_equal(cm,metrics["confusion_matrix"])
        history=pd.read_csv(art/"runs"/row.run_id/"history.csv")
        assert abs(history.val_macro_f1.max()-row.best_val_macro_f1)<1e-12
    receipt={"completed_runs":len(validation),"test_evaluations":len(test),"test_reviews":len(truth),
             "checks":["text disjointness","train/test label alignment","equal search coverage",
                       "validation-only selections","parameter consistency","independent accuracy and Macro-F1 arithmetic",
                       "confusion matrix reconciliation","checkpoint epoch histories","selection saved before test predictions"]}
    (art/"verification.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
