"""Package source, measured results, weights and submission documents locally."""
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    art=ROOT/"artifacts"
    results=pd.read_csv(art/"test_results.csv")
    selected=json.loads((art/"selection_before_test.json").read_text())
    rows=["## Measured results","","All models use the same 48,257 unique held-out reviews. Selections use validation Macro-F1.","",
          "| Cell | Fixed accuracy | Fixed Macro-F1 | Selected setting | Selected accuracy | Selected Macro-F1 |",
          "| --- | --- | --- | --- | --- | --- |"]
    for v,name in [("vanilla","Vanilla"),("cifg","CIFG"),("wmc","WMC")]:
        fixed=results[(results.variant==v)&results.comparison.str.contains("fixed")].iloc[0]
        tuned=results[results.run_id==selected[v]].iloc[0]
        rows.append(f"| {name} | {fixed.test_accuracy:.2%} | {fixed.test_macro_f1:.4f} | {tuned.setting} | {tuned.test_accuracy:.2%} | {tuned.test_macro_f1:.4f} |")
    rows.extend(["","These are executed, reduced-data, single-seed results. They do not establish a universal architectural ranking.",""])
    readme_path=ROOT/"README.md"
    readme=readme_path.read_text(encoding="utf-8")
    begin="<!-- measured-results:start -->"
    end="<!-- measured-results:end -->"
    block=begin+"\n"+"\n".join(rows)+end+"\n\n"
    if begin in readme:
        before,rest=readme.split(begin,1);_,after=rest.split(end,1)
        readme=before+block+after.lstrip("\n")
    else:
        readme=readme.replace("## Run locally",block+"## Run locally",1)
    readme_path.write_text(readme,encoding="utf-8")
    shutil.copyfile(ROOT/"data/processed/data_audit.json",art/"data_audit.json")
    archive=ROOT/"output/Drug_Review_Project.zip"
    files=[]
    for directory in ["src","tests","scripts","artifacts","output"]:
        files.extend(p for p in (ROOT/directory).rglob("*") if p.is_file())
    files.extend(ROOT/name for name in ["README.md","pyproject.toml","requirements-lock.txt","run_project.ps1",".gitignore",".gitattributes"])
    files.extend(ROOT/"data/processed"/name for name in ["vocabulary.json","data_audit.json","split_manifest.csv"])
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            if p.suffix==".zip" or "__pycache__" in p.parts or not p.is_file():
                continue
            z.write(p,p.relative_to(ROOT))
    print(f"Packaged {archive.name}: {archive.stat().st_size/1024/1024:.1f} MB")


if __name__=="__main__":
    main()
