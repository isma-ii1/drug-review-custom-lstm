import pandas as pd

from drug_review.data import build_data, read_source


def test_source_conflicts_are_dropped(tmp_path):
    rows=pd.DataFrame({"uniqueID":[1,2,3,4,5],"review":["same text","SAME TEXT","good text","good text","   "],
                       "rating":[1,9,9,10,5]})
    path=tmp_path/"source.csv";rows.to_csv(path,index=False)
    clean,audit=read_source(path)
    assert list(clean.normalized)==["good text"]
    assert audit["conflicting_rows_removed"]==2
    assert audit["same_label_duplicates_removed"]==1
    assert audit["invalid_required_rows"]==1


def test_no_text_overlap_or_vocabulary_leakage(tmp_path):
    rows=[]
    for label,rating in enumerate([2,5,9]):
        for i in range(20):
            rows.append({"uniqueID":len(rows),"review":f"train group {label} example {i}","rating":rating})
    # Matching source text must be removed even though ID differs.
    test=pd.DataFrame([{"uniqueID":100,"review":rows[0]["review"],"rating":2},
        {"uniqueID":101,"review":"exclusiveholdoutword only","rating":9}])
    train_path=tmp_path/"train.csv";test_path=tmp_path/"test.csv"
    pd.DataFrame(rows).to_csv(train_path,index=False);test.to_csv(test_path,index=False)
    out=tmp_path/"processed"
    audit=build_data(train_path,test_path,out,train_limit=18,val_limit=6,vocab_size=100,max_length=20)
    import json
    vocab=json.loads((out/"vocabulary.json").read_text())
    manifest=pd.read_csv(out/"split_manifest.csv")
    assert "exclusiveholdoutword" not in vocab
    assert not manifest.text_sha256.duplicated().any()
    assert audit["train_source"]["rows_overlapping_raw_test_removed"]==1
    assert audit["splits"]["train"]["rows"]==18
