"""Build the five-page report and ten-slide defense from executed results."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"
FIG=OUT/"figures"
ART=ROOT/"artifacts"
DATA=ROOT/"data/processed"
LABELS=["Negative","Neutral","Positive"]
NAMES={"vanilla":"Vanilla","cifg":"CIFG","wmc":"WMC"}
PALETTE={"vanilla":"#2563A6","cifg":"#B66A00","wmc":"#6B7343"}
INK=colors.HexColor("#172E46")
BLUE=colors.HexColor("#2563A6")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent(v):
    return f"{100*v:.2f}%"


def charts(audit,validation,test,selection):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"axes.spines.top":False,
        "axes.spines.right":False,"axes.titleweight":"bold","figure.facecolor":"white"})
    counts=[audit["splits"]["train"]["class_counts"][k] for k in LABELS]
    fig,axes=plt.subplots(1,2,figsize=(10,2.6),layout="constrained")
    bars=axes[0].bar(LABELS,counts,color="#2563A6",width=.6)
    axes[0].bar_label(bars,labels=[f"{v:,}\n({v/sum(counts):.1%})" for v in counts],padding=3,fontsize=11)
    axes[0].set_ylim(0,max(counts)*1.3)
    axes[0].set_ylabel("Training reviews")
    axes[0].set_title("Rating-derived class balance")
    lengths=np.load(DATA/"training_lengths.npy")
    axes[1].hist(lengths,bins=np.arange(0,221,10),color="#2563A6",rwidth=.95)
    axes[1].axvline(160,color="#9A5A00",linestyle="--",label="160-token limit")
    axes[1].set_xlabel("Tokens per training review")
    axes[1].set_ylabel("Reviews")
    axes[1].set_title("Review lengths")
    axes[1].legend(frameon=False,fontsize=11)
    fig.savefig(FIG/"eda.png",dpi=200); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,2.7),layout="constrained")
    for v,marker,line_style in [("vanilla","o","-"),("cifg","s","--"),("wmc","^",":")]:
        history=pd.read_csv(ART/"runs"/(v+"_fixed")/"history.csv")
        axes[0].plot(history.epoch,history.val_macro_f1,marker=marker,linestyle=line_style,label=NAMES[v],color=PALETTE[v])
        axes[1].plot(history.epoch,history.train_weighted_loss,marker=marker,linestyle=line_style,label=NAMES[v],color=PALETTE[v])
    axes[0].set(xlabel="Epoch",ylabel="Validation Macro-F1",ylim=(0,1))
    axes[1].set(xlabel="Epoch",ylabel="Training weighted cross-entropy")
    for ax in axes:
        ax.legend(frameon=False,fontsize=11); ax.grid(axis="y",alpha=.2)
    fig.savefig(FIG/"learning_curves.png",dpi=200); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,2.7),layout="constrained")
    for ax,(v,name) in zip(axes,NAMES.items()):
        evaluation=read_json(ART/"evaluation"/(selection[v]+".json"))
        cm=np.array(evaluation["confusion_matrix"])
        normalized=cm/cm.sum(1,keepdims=True)
        ax.imshow(normalized,vmin=0,vmax=1,cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f"{cm[i,j]:,}\n{normalized[i,j]:.0%}",ha="center",va="center",fontsize=11,
                        color="white" if normalized[i,j]>.55 else "#172E46")
        ax.set_xticks(range(3),["Neg","Neu","Pos"])
        ax.set_yticks(range(3),["Neg","Neu","Pos"])
        ax.set_title(name+" selected"); ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True class")
    fig.savefig(FIG/"confusion_matrices.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3),layout="constrained")
    x=np.arange(3); width=.32
    fixed=[]; tuned=[]
    for v in NAMES:
        fixed.append(float(test[(test.variant==v)&test.comparison.str.contains("fixed")].iloc[0].test_macro_f1))
        tuned.append(float(test[test.run_id==selection[v]].iloc[0].test_macro_f1))
    ax.bar(x-width/2,fixed,width,label="Fixed settings",color="#2563A6")
    bars=ax.bar(x+width/2,tuned,width,label="Selected on validation",color="#B66A00")
    ax.bar_label(bars,labels=[f"{v:.3f}" for v in tuned],padding=3)
    ax.set_xticks(x,list(NAMES.values())); ax.set_ylim(0,1); ax.set_ylabel("Test Macro-F1")
    ax.legend(frameon=False); ax.grid(axis="y",alpha=.2)
    fig.savefig(FIG/"test_comparison.png",dpi=200); plt.close(fig)


def error_analysis(selection):
    records=pd.read_csv(DATA/"test_records.csv")
    summaries=[]; examples=[]
    from drug_review.data import tokenize
    lengths=records.review.map(lambda s:len(tokenize(s)))
    negation=records.review.str.contains(r"\b(?:not|never|no)\b|n't\b",case=False,regex=True)
    from sklearn.metrics import f1_score,accuracy_score
    for v,run_id in selection.items():
        pred=pd.read_csv(ART/"evaluation"/(run_id+"_predictions.csv"))
        assert (records.label.to_numpy()==pred.true_label.to_numpy()).all()
        for name,mask in {"all":pd.Series(True,index=records.index),"negation":negation,
                          "over_160_tokens":lengths>160,"neutral":records.label==1}.items():
            y=pred.loc[mask,"true_label"]; p=pred.loc[mask,"predicted_label"]
            summaries.append({"variant":v,"slice":name,"n":len(y),"accuracy":accuracy_score(y,p),
                "macro_f1":f1_score(y,p,labels=[0,1,2],average="macro",zero_division=0)})
        if v=="vanilla":
            for true_class in [1,0]:
                bad=pred.index[(pred.true_label==true_class)&(pred.true_label!=pred.predicted_label)]
                if len(bad):
                    i=int(bad[0]); r=records.iloc[i]
                    examples.append({"uniqueID":int(r.uniqueID),"true":LABELS[true_class],
                        "predicted":LABELS[int(pred.iloc[i].predicted_label)],
                        "excerpt":" ".join(html.unescape(str(r.review)).split()).strip('\"')[:220]})
    pd.DataFrame(summaries).to_csv(ART/"error_slices.csv",index=False)
    (ART/"error_examples.json").write_text(json.dumps(examples,indent=2),encoding="utf-8")
    return pd.DataFrame(summaries),examples


def build_report(audit,validation,test,selection,slices,examples,repo_url):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle",fontName="Helvetica-Bold",fontSize=20,leading=23,textColor=INK,spaceAfter=10))
    styles.add(ParagraphStyle(name="ReportH",fontName="Helvetica-Bold",fontSize=12,leading=15,textColor=BLUE,spaceBefore=10,spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportBody",fontName="Helvetica",fontSize=10,leading=13.5,spaceAfter=7))
    styles.add(ParagraphStyle(name="SmallNote",fontName="Helvetica",fontSize=8.2,leading=10.5,textColor=colors.HexColor("#445466"),spaceAfter=6))
    styles.add(ParagraphStyle(name="Cell",fontName="Helvetica",fontSize=8.5,leading=10.5))
    story=[]
    def p(s,kind="ReportBody"):
        story.append(Paragraph(s,styles[kind]))
    def h(s):p(s,"ReportH")
    def table(rows,widths=None):
        converted=[[Paragraph(html.escape(str(v)),styles["Cell"]) for v in row] for row in rows]
        t=Table(converted,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E8EFF6")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),
            ("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),("LINEBELOW",(0,0),(-1,0),.6,BLUE),
            ("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#D5DDE6"))]))
        story.append(t);story.append(Spacer(1,7))
    def figure(name,height):story.append(Image(str(FIG/name),width=495,height=height));story.append(Spacer(1,4))
    def newpage():story.append(PageBreak())
    selected=test[test.run_id.isin(selection.values())].set_index("variant")
    fixed=test[test.comparison.str.contains("fixed")].set_index("variant")
    best_variant=max(NAMES,key=lambda v:float(validation[validation.run_id==selection[v]].iloc[0].best_val_macro_f1))
    best=selected.loc[best_variant]
    baseline=read_json(ART/"majority_baseline.json")
    # Page 1
    p("Drug Review Classification<br/>with Custom LSTM Cells","ReportTitle")
    p("Universite Internationale de Rabat - Advanced NLP - 2026/2027<br/>Ismail Ighmour - Professor Younes Moukafih","SmallNote")
    h("1. Objective and principal findings")
    p(f"We compare Vanilla LSTM, Coupled Input-Forget Gate LSTM (CIFG), and Working Memory Connections (WMC) on rating-derived drug review classification. All cells follow the course appendix and use manual PyTorch nn.Linear transformations. The validation-selected {NAMES[best_variant]} configuration achieves {percent(best.test_accuracy)} accuracy and {best.test_macro_f1:.4f} Macro-F1 on 48,257 cleaned test reviews. This is one reduced-data, single-seed experiment, rather than a general ranking of LSTM designs.")
    h("2. Dataset and target definition")
    p("The supplied Drugs.com CSVs contain 161,297 training and 53,766 test rows. Only review text enters the classifier. The rating constructs the label and never enters the input. Drug name, condition, date, usefulCount, and uniqueID are excluded from predictive features. The three-class mapping follows the supplied draft report and is a project choice, not a threshold prescribed by the guidelines.")
    table([["Rating","Class","Meaning"],["1-4","Negative (0)","Lower reported satisfaction"],["5-6","Neutral (1)","Intermediate reported satisfaction"],["7-10","Positive (2)","Higher reported satisfaction"]],[55,110,330])
    figure("eda.png",129)
    p("Figure 1. The 15,000-review training subset preserves the source class proportions. Positive reviews dominate (66.54%), while Neutral accounts for only 8.85%. Median length is 86 tokens and the 95th percentile is 148 tokens.","SmallNote")
    h("3. Quality checks and duplicate leakage")
    p("Review IDs are unique in both source files. Condition is missing in 899 training and 295 test rows, which does not prevent a text-only task. After HTML decoding, case normalization, and whitespace cleanup, 49,012 training rows repeat an earlier review text. The test file also contains 5,504 repeated text rows. Identical texts crossing partitions could reward memorization, so we remove them before modeling.")
    # Page 2
    newpage();h("4. Preprocessing and shared data protocol")
    p("Each source file is cleaned independently. We discard invalid or token-empty reviews, remove all texts with conflicting three-class labels within that file, and keep one copy of each remaining text. We then exclude every cleaned training text that appears anywhere in the raw test file, regardless of its rating. This removes 32,111 candidate training rows. Test labels do not enter vocabulary construction, weighting, optimization, or model selection.")
    table([["Stage / split","Rows","Selection rule"],["Eligible source training","80,159","After cleaning and excluding test texts"],["Training pool / validation pool","64,127 / 16,032","Stratified 80/20 split, seed 42"],["Executed training / validation","15,000 / 3,000","Fixed stratified subsets, seeds 42 / 43"],["Evaluation test set","48,257","All unique valid, unambiguous test texts"]],[145,90,260])
    p("The guidelines permit reduced training data when compute is limited. All 12 runs use these exact same partitions, saved by ID and text hash. The 12,000-entry vocabulary, including PAD and UNK, is fitted only to the executed training subset. Tokenization retains negation and contractions. Each review contributes its first 160 tokens, with dynamic batch padding and true-length state masking. Only 0.46% of training and 0.52% of test reviews are truncated. Test unknown-token rate is 1.05%.")
    h("5. Recurrent architectures and appendix equations")
    p("For each gate q, define z_q = W_xq x_t + W_hq h_(t-1) + b_q. Vanilla uses i = sigmoid(z_i), f = sigmoid(z_f), o = sigmoid(z_o), g = tanh(z_g), c_t = f*c_(t-1) + i*g, and h_t = o*tanh(c_t). Multiplication is elementwise. We concatenate gate matrices for efficiency while preserving the same nn.Linear operations and equations.")
    p("CIFG removes the independent input gate and sets i = 1 - f. WMC adds tanh(W_ci c_(t-1)) to z_i, tanh(W_cf c_(t-1)) to z_f, and tanh(W_co c_t) to z_o. The output gate therefore reads the updated memory c_t. WMC uses full learned memory projection matrices, with no extra memory bias. All models share an embedding, the final valid hidden state, dropout, and a three-logit linear output.")
    rows=[["Cell","Recurrent parameters","Total parameters"]]
    for v in NAMES:
        r=fixed.loc[v];rows.append([NAMES[v],f"{r.recurrent_parameters:,}",f"{r.total_parameters:,}"])
    table(rows,[145,175,175])
    p("For embedding dimension E=64 and hidden size H=32, Vanilla has 4H(E+H+1) recurrent parameters, CIFG has 3H(E+H+1), and WMC adds 3*H^2 to Vanilla. CIFG reduces the recurrent cell by 25%. The shared 768,000-parameter embedding dominates total size, so the reduction in the complete classifier is much smaller.","SmallNote")
    # Page 3
    newpage();h("6. Training controls and hyperparameter study")
    p("Each run starts from seed 42 with 64-dimensional embeddings and a 32-dimensional hidden state. Embeddings start from a zero-mean normal distribution (standard deviation 0.1), with a fixed zero padding row. Input gate blocks use Xavier initialization, recurrent gate blocks use orthogonal initialization, and the forget-gate bias starts at 1. Dropout is 0.2 on embeddings and the final representation. The loss is class-weighted cross-entropy, with class weight N/(3*n_class) computed from training labels only. All runs clip the global gradient norm at 1.")
    p("The fixed configuration uses AdamW, learning rate 0.003, batch size 256, and weight decay 0.01. Each of the other settings changes one factor: learning rate 0.001, batch size 128, or Adam instead of AdamW. Weight decay remains 0.01, so the optimizer comparison includes their different regularization semantics. Each architecture receives the same four-setting search budget, up to six epochs, with early stopping after three epochs without validation Macro-F1 improvement. Checkpoints maximize validation Macro-F1.")
    pivot=validation.pivot(index="setting",columns="variant",values="best_val_macro_f1")
    rows=[["Setting","Vanilla val F1","CIFG val F1","WMC val F1"]]
    for setting in ["fixed","lr_0.001","batch_128","adam"]:
        rows.append([setting]+[f"{pivot.loc[setting,v]:.4f}" for v in NAMES])
    table(rows,[165,110,110,110])
    p("Table 2. Best validation Macro-F1 for each run, from actual epoch histories. The checkpoint epoch and elapsed training time are saved with each experiment. The same maximum epoch budget does not imply identical gradient update counts when batch sizes differ.","SmallNote")
    figure("learning_curves.png",143)
    p("Figure 2. Fixed-setting learning curves. Left: validation Macro-F1. Right: weighted training loss. Training loss and unweighted validation loss use different reductions and should not be compared as equivalent quantities.","SmallNote")
    env=read_json(ART/"environment.json")
    p(f"Runs execute on CPU with two PyTorch threads (PyTorch {env['torch']}, Python {env['python']}). Timing includes training and per-epoch validation, but excludes preprocessing and final test inference. Exact package versions, source hashes, split manifests, configurations, checkpoints, and raw prediction files support reproduction.","SmallNote")
    # Page 4
    newpage();h("7. Held-out evaluation and model comparison")
    p("Macro-F1 is the primary measure because it weights each class equally despite imbalance. Accuracy, per-class precision/recall/F1, confusion matrices, and computational cost provide complementary evidence. We freeze each architecture's validation winner before test scoring. Fixed-setting results isolate the cell change; selected-setting results compare equal-budget validation searches. No model or checkpoint is chosen by test performance.")
    rows=[["Model","Fixed acc.","Fixed F1","Selected setting","Selected acc.","Selected F1"]]
    for v in NAMES:
        r=selected.loc[v];f=fixed.loc[v]
        rows.append([NAMES[v],percent(f.test_accuracy),f"{f.test_macro_f1:.4f}",r.setting,percent(r.test_accuracy),f"{r.test_macro_f1:.4f}"])
    table(rows,[65,72,65,140,80,73])
    p(f"A training-majority Positive classifier scores {percent(baseline['test_accuracy'])} accuracy and {baseline['test_macro_f1']:.4f} Macro-F1. Neural results should be judged against both, since high accuracy alone can hide minority-class failure.","SmallNote")
    rows=[["Selected model","Negative P / R / F1","Neutral P / R / F1","Positive P / R / F1"]]
    for v in NAMES:
        evaluation=read_json(ART/"evaluation"/(selection[v]+".json"))
        rows.append([NAMES[v]]+[" / ".join(f"{evaluation['classification_report'][label][k]:.3f}" for k in ["precision","recall","f1-score"]) for label in LABELS])
    table(rows,[80,138,138,139])
    figure("confusion_matrices.png",137)
    p("Figure 3. Selected models on the same test set. Counts and row percentages show which true classes each model confuses. Support: 12,093 Negative, 4,334 Neutral, and 31,830 Positive reviews.","SmallNote")
    rows=[["Architecture","Fixed min / epochs","Search total (min)","Selected epoch"]]
    for v in NAMES:
        rows.append([NAMES[v],f"{fixed.loc[v].training_seconds/60:.2f} / {int(fixed.loc[v].epochs_run)}",f"{validation[validation.variant==v].training_seconds.sum()/60:.2f}",int(selected.loc[v].best_epoch)])
    table(rows,[115,130,140,110])
    # Page 5
    newpage();h("8. Interpretation, errors, and limitations")
    cifg_delta=100*(fixed.loc["cifg"].test_macro_f1-fixed.loc["vanilla"].test_macro_f1)
    wmc_delta=100*(fixed.loc["wmc"].test_macro_f1-fixed.loc["vanilla"].test_macro_f1)
    p(f"At fixed settings, CIFG changes test Macro-F1 by {cifg_delta:+.2f} percentage points relative to Vanilla, while WMC changes it by {wmc_delta:+.2f} points. CIFG saves 3,104 recurrent parameters, corresponding to only {3104/780515:.2%} of the full Vanilla classifier. These results measure one initialization and one training subset. Small performance differences cannot establish architectural superiority without repeated seeds and a comparison uncertainty analysis.")
    adam=validation[validation.setting=="adam"]
    adamw=validation[validation.setting=="fixed"]
    p(f"The optimizer study yields validation Macro-F1 ranges of {adam.best_val_macro_f1.min():.3f}-{adam.best_val_macro_f1.max():.3f} for Adam and {adamw.best_val_macro_f1.min():.3f}-{adamw.best_val_macro_f1.max():.3f} for fixed AdamW. Weight decay 0.01 has different semantics in these optimizers. The weak Adam runs therefore describe this regularization setting, rather than proving that Adam is generally unsuitable. A follow-up should tune optimizer-specific regularization on validation data.")
    p("CIFG trades independent write control for fewer recurrent parameters. WMC gives gates direct access to cell memory at added cost. Their observed predictive results must be considered alongside these parameter counts and measured timing, rather than assuming that the more elaborate mechanism will perform better. The selected overall candidate follows validation Macro-F1, even if test rankings differ.")
    best_classes=read_json(ART/"evaluation"/(selection[best_variant]+".json"))["classification_report"]
    weakest=min(LABELS,key=lambda label:best_classes[label]["f1-score"])
    p(f"For the overall validation-selected candidate, {weakest} is the weakest class (F1 {best_classes[weakest]['f1-score']:.3f}). This class-level result explains why overall accuracy must be read alongside Macro-F1. The confusion matrices identify the relevant destination classes without attributing the mistakes to an unverified linguistic cause.")
    for example in examples[:2]:
        safe=html.escape(example["excerpt"])
        p(f"Illustrative Vanilla error (review ID {example['uniqueID']}): true {example['true']}, predicted {example['predicted']}. Excerpt: &quot;{safe}...&quot;","SmallNote")
    p("These deterministic examples illustrate individual mistakes and do not prove a systematic cause. Saved error-slice tables compare all reviews, a heuristic negation subset, reviews longer than 160 tokens, and Neutral reviews. Subset Macro-F1 can include absent classes, so the Neutral-only slice is interpreted through its accuracy (Neutral recall), not its three-class Macro-F1.")
    p("The principal limitations are the small training subset, a compact hidden state, a limited search, and one seed. Ratings are a noisy proxy for textual sentiment. Our strict text-deduplicated population differs from the raw official split, so its scores are not directly comparable with published results on all raw rows. Exact text deduplication does not detect paraphrases, shared templates, patient identity, or all potential group leakage. Random source splits do not test temporal or unseen-drug transfer.")
    h("9. Improvements and conclusion")
    p("Next experiments should retain the frozen test protocol while using the full eligible training pool, multiple seeds, and a broader validation search. Pretrained embeddings or attention may improve representations, while additional class-balance strategies and error annotation may clarify Neutral performance. Longer context should be tested only where the small truncated subset warrants its extra cost. The delivered project establishes a controlled comparison of the three required custom cells with measured predictive and computational results.")
    h("References and reproducibility")
    p("[1] Y. Moukafih. Lab Project on Recurrent Neural Networks, Advanced NLP, UIR, 2026/2027, pp. 1-2. Supplied guidelines and appendix.<br/>[2] Supplied Drugs.com training and test CSVs. Course dataset reference: archive.ics.uci.edu/dataset/462.<br/>[3] K. Greff et al. LSTM: A Search Space Odyssey. arxiv.org/abs/1503.04069.<br/>[4] F. Landi et al. Working Memory Connections for LSTM. arxiv.org/abs/2109.00020.<br/>[5] PyTorch 2.15 Adam and AdamW API documentation: docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html and torch.optim.AdamW.html.","SmallNote")
    p(f"PyTorch repository: <link href='{repo_url}' color='#2563A6'>{repo_url}</link><br/>Results: artifacts/validation_results.csv, test_results.csv, evaluation/, error_slices.csv. Split and preprocessing audit: data/processed/data_audit.json and split_manifest.csv.","SmallNote")
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor("#D5DDE6"));c.line(50,42,545,42)
        c.setFont("Helvetica",8);c.setFillColor(INK)
        c.drawString(50,29,"UIR - Advanced NLP - Drug Review Classification")
        c.drawRightString(545,29,str(doc.page))
    doc=SimpleDocTemplate(str(OUT/"Drug_Review_Final_Report.pdf"),pagesize=(595.28,841.89),
        leftMargin=50,rightMargin=50,topMargin=38,bottomMargin=52,
        title="Drug Review Classification with Custom LSTM Cells",author="Ismail Ighmour")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def slide_content(audit,validation,test,selection,repo_url):
    selected=test[test.run_id.isin(selection.values())].set_index("variant")
    fixed=test[test.comparison.str.contains("fixed")].set_index("variant")
    best_v=max(NAMES,key=lambda v:float(validation[validation.run_id==selection[v]].iloc[0].best_val_macro_f1))
    compare_rows=[[NAMES[v],percent(fixed.loc[v].test_accuracy),f"{fixed.loc[v].test_macro_f1:.4f}",selected.loc[v].setting,f"{selected.loc[v].test_macro_f1:.4f}"] for v in NAMES]
    parameter_rows=[[NAMES[v],f"{fixed.loc[v].recurrent_parameters:,}",f"{fixed.loc[v].total_parameters:,}"] for v in NAMES]
    search=validation.pivot(index="setting",columns="variant",values="best_val_macro_f1")
    search_rows=[[s]+[f"{search.loc[s,v]:.4f}" for v in NAMES] for s in ["fixed","lr_0.001","batch_128","adam"]]
    return [
        {"title":"Drug review classification","subtitle":"Vanilla LSTM, CIFG and Working Memory Connections",
         "lines":["Ismail Ighmour", "Universite Internationale de Rabat", "Advanced NLP, 2026/2027"],"seconds":35,
         "notes":"Our task is to infer rating-derived satisfaction from patient-written review text. We compare the standard LSTM gate design with two appendix variants. The purpose is to measure the effect of the memory mechanism under one common experimental protocol."},
        {"title":"Task and labels","lines":["Input: review text only", "Ratings 1-4: Negative", "Ratings 5-6: Neutral", "Ratings 7-10: Positive", "Ratings define targets and never enter the model"],"seconds":50,
         "notes":"The supplied dataset contains a numerical satisfaction rating and a review. We follow the draft report's explicit three-class mapping. The guidelines require classification but do not prescribe these thresholds. The labels are a proxy for satisfaction, and a mixed review may not align neatly with its numerical rating. We exclude metadata and rating from inputs to avoid direct target leakage."},
        {"title":"Class imbalance and review lengths","image":"eda.png","lines":["66.54% Positive and 8.85% Neutral in training", "160 tokens retain complete text for 99.54% of training reviews"],"seconds":60,
         "notes":"The raw files have 161,297 training rows and 53,766 test rows. The executed training subset is strongly imbalanced. This is why accuracy alone is insufficient and Macro-F1 is our primary measure. Review lengths support a 160-token limit: fewer than one percent of reviews exceed it. We use class-weighted loss with weights derived only from training data."},
        {"title":"Duplicate leakage and shared partitions","lines":["32,111 cleaned training texts also appear in raw test", "Exclude these texts before splitting training data", "15,000 training, 3,000 validation, 48,257 test reviews", "Same partitions and train-only vocabulary for all models"],"seconds":60,
         "notes":"Deduplication is one of the most consequential findings. A random split without checking text overlap could reward memorization. We normalize HTML, case and whitespace, remove within-file conflicting labels, and retain unique texts. We give the supplied test source priority and exclude every matching training text. We stratify the eligible training pool, then use fixed subsets permitted by the guidelines. The test set remains all cleaned unique eligible test reviews. This protocol changes the evaluation population, which limits comparison to raw-split published scores."},
        {"title":"Three custom recurrent cells","lines":["Vanilla: independent input, forget and output gates", "CIFG: input gate i = 1 - f", "WMC: tanh memory projections enter the gates", "WMC output gate reads the updated memory c_t", "All linear operations use nn.Linear"],"seconds":85,
         "notes":"All models share a 64-dimensional embedding, a 32-dimensional hidden state and the same classification head. Vanilla computes independent input, forget and output gates and a candidate. CIFG ties the input gate to the complement of the forget gate, reducing the cell's parameters. WMC adds learned full-matrix nonlinear projections of the previous memory into the input and forget gates. Its output gate uses the updated current memory. We implement every cell manually with nn.Linear and elementwise operations. We never use nn.LSTM. Masking preserves the final valid hidden state rather than a state produced by padding. Direct equation, gradient and padding checks passed."},
        {"title":"Parameter cost","headers":["Cell","Recurrent parameters","Total parameters"],"rows":parameter_rows,"lines":["CIFG reduces recurrent parameters by 25%", "The shared embedding has 768,000 parameters"],"seconds":60,
         "notes":"Recurrent and total parameter counts answer different questions. CIFG removes one set of gate transformations, which saves exactly a quarter of the recurrent parameters at the same dimensions. WMC adds three full hidden-to-hidden memory matrices. Because the embedding dominates this small classifier, cell savings translate into a much smaller percentage reduction for the complete model. This is why the report includes both counts and actual training times."},
        {"title":"Hyperparameter study","headers":["Setting","Vanilla val F1","CIFG val F1","WMC val F1"],"rows":search_rows,
         "lines":["Fixed: AdamW, LR 0.003, batch 256", "One-factor changes: LR 0.001, batch 128, Adam", "Six-epoch maximum and validation-based early stopping"],"seconds":65,
         "notes":"Each architecture receives the same four configurations. The fixed comparison changes only the recurrent cell. The additional settings each change one factor, so their effects remain interpretable. We keep dropout at 0.2 and weight decay at 0.01. Adam and AdamW treat weight decay differently, so that is part of the optimizer comparison. All runs use seed 42, norm clipping at one and identical class weights. We select the checkpoint by validation Macro-F1 and freeze the configuration selections before evaluating test predictions."},
        {"title":"Held-out performance","headers":["Cell","Fixed accuracy","Fixed F1","Selected setting","Selected F1"],"rows":compare_rows,
         "lines":["All scores use the same 48,257 test reviews", "Selected settings follow validation Macro-F1"],"seconds":80,
         "notes":"This table separates the fixed-setting architectural comparison from each architecture's validation-selected result. Macro-F1 gives every class the same weight. The report also includes precision and recall by class and full confusion matrices, since a model can score high accuracy by favoring Positive. The differences here describe a compact model trained on a reduced subset with one seed. They do not establish a universally superior cell. Test performance does not determine our chosen model."},
        {"title":"Errors and limits","image":"confusion_matrices.png","lines":["Neutral ratings cover mixed experiences", "Single seed, reduced training data and compact hidden state", "Exact deduplication leaves paraphrases and group risks"],"seconds":60,
         "notes":"The confusion matrices show which true classes the selected models confuse. Neutral is a small class and its rating definition can cover mixed language. The project saves deterministic error examples and exploratory negation and length slices. Those subsets do not prove causal explanations for errors. We use one seed and a limited search, so small gaps may reflect optimization variation. Exact text deduplication cannot eliminate paraphrases or shared patient identity, and random splitting cannot establish temporal generalization."},
        {"title":"Conclusion and next experiments","lines":[f"Validation-selected candidate: {NAMES[best_v]} ({selection[best_v]})",
            "Custom cells support a controlled, measured comparison", "Next: full training pool, repeated seeds and broader tuning", "Pretrained embeddings and attention are useful extensions"],"seconds":45,
         "notes":f"Our main contribution is a reproducible comparison that follows the prescribed cell equations and prevents exact text leakage across partitions. The overall candidate follows validation Macro-F1. The strongest next experiment is to retain the frozen protocol and increase training coverage with repeated seeds. A broader validation search and pretrained embeddings may improve quality. The complete PyTorch source, experiment logs, parameter counts and report are available at {repo_url}."},
    ]


def build_slides(slides,repo_url):
    sections=[]
    for i,s in enumerate(slides,1):
        elements=[f"<h1>{html.escape(s['title'])}</h1>"]
        if s.get("subtitle"):elements.append(f"<p class='subtitle'>{html.escape(s['subtitle'])}</p>")
        if s.get("image"):elements.append(f"<img src='figures/{s['image']}' alt='{html.escape(s['title'])}'>")
        if s.get("rows"):
            elements.append("<table><thead><tr>"+"".join(f"<th>{html.escape(str(v))}</th>" for v in s["headers"])+"</tr></thead><tbody>"+
                "".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in row)+"</tr>" for row in s["rows"])+"</tbody></table>")
        elements.append("<div class='lines'>"+"".join(f"<p>{html.escape(line)}</p>" for line in s.get("lines",[]))+"</div>")
        elements.append(f"<aside class='notes'><b>{s['seconds']} seconds</b><p>{html.escape(s['notes'])}</p></aside><footer>{i} / {len(slides)}</footer>")
        sections.append(f"<section class='slide' id='slide-{i}'>"+"".join(elements)+"</section>")
    page="""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Drug Review Defense</title>
<style>*{box-sizing:border-box}body{margin:0;background:#172e46;font-family:Arial,sans-serif;color:#172e46}.slide{display:none;position:relative;width:min(100vw,160vh);aspect-ratio:16/9;background:white;margin:auto;padding:4.5%;overflow:hidden}h1{font-size:clamp(24px,3.5vw,52px);margin:0 0 3%}p{font-size:clamp(16px,2.1vw,30px);line-height:1.35;margin:1.5% 0}.subtitle{color:#2563a6}img{width:100%;max-height:45%;object-fit:contain}table{border-collapse:collapse;width:100%;font-size:clamp(14px,1.7vw,24px);margin-bottom:2%}th,td{text-align:left;padding:1.1%;border-bottom:1px solid #cad5df}th{background:#e8eff6}.notes{display:none}footer{position:absolute;bottom:3%;right:5%;font-size:16px;color:#52677d}nav{display:flex;justify-content:center;gap:1em;padding:12px}button{padding:.6em 1em;cursor:pointer}.presenter .notes{display:block;background:#eef3f7;padding:1%;font-size:14px}.presenter .notes p{font-size:14px}.presenter .slide{overflow:auto;aspect-ratio:auto;min-height:80vh}.presenter img{max-height:24vh}body:not(.presenter) .slide.active{display:block}.presenter .slide.active{display:block}@media print{body{background:white}.slide{display:block!important;width:100%;height:100vh;page-break-after:always}nav,.notes{display:none!important}@page{size:landscape;margin:0}}</style>
<body>"""+"".join(sections)+"""<nav><button id='previous'>Previous</button><button id='next'>Next</button><button id='notes'>Speaker notes (N)</button><button id='full'>Fullscreen (F)</button></nav>
<script>let index=0;const slides=[...document.querySelectorAll('.slide')];function show(n){index=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===index));history.replaceState(null,'','#slide-'+(index+1))}function notes(){document.body.classList.toggle('presenter')}function full(){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}document.querySelector('#next').onclick=()=>show(index+1);document.querySelector('#previous').onclick=()=>show(index-1);document.querySelector('#notes').onclick=notes;document.querySelector('#full').onclick=full;document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}if(e.key.toLowerCase()==='n')notes();if(e.key.toLowerCase()==='f')full();if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1)});show(Number(location.hash.replace('#slide-',''))-1||0);</script></body></html>"""
    (OUT/"Drug_Review_Defense.html").write_text(page,encoding="utf-8")
    notes=["# Defense speaker notes","","Total planned time: 10 minutes (600 seconds).","",
           "Open Drug_Review_Defense.html and use Left/Right to navigate, N for notes, F for fullscreen.",""]
    for i,s in enumerate(slides,1):
        notes.extend([f"## Slide {i}: {s['title']} ({s['seconds']} seconds)","",s["notes"],""])
    notes.extend(["## Sources","","Course guidelines and appendix, supplied training/test CSVs, artifacts/validation_results.csv and artifacts/test_results.csv.",
                  "Greff et al.: https://arxiv.org/abs/1503.04069", "Landi et al.: https://arxiv.org/abs/2109.00020",repo_url])
    (OUT/"Speaker_Notes.md").write_text("\n".join(notes),encoding="utf-8")


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--repo-url",required=True);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    audit=read_json(DATA/"data_audit.json")
    validation=pd.read_csv(ART/"validation_results.csv")
    test=pd.read_csv(ART/"test_results.csv")
    selection=read_json(ART/"selection_before_test.json")
    assert len(validation)==12 and len(selection)==3
    expected_sizes={"train":15000,"validation":3000,"test":48257}
    if (any(audit["splits"][k]["rows"]!=v for k,v in expected_sizes.items())
            or audit["max_length"]!=160 or audit["vocab_size"]!=12000
            or not (validation.embedding_dim==64).all() or not (validation.hidden_size==32).all()
            or not (validation.seed==42).all()):
        raise ValueError("The written methods describe the supplied default study. Update the narrative for a changed protocol.")
    charts(audit,validation,test,selection)
    slices,examples=error_analysis(selection)
    build_report(audit,validation,test,selection,slices,examples,args.repo_url)
    slides=slide_content(audit,validation,test,selection,args.repo_url)
    assert sum(s["seconds"] for s in slides)==600
    build_slides(slides,args.repo_url)
    lengths=np.load(DATA/"training_lengths.npy")
    content={"slides":slides,"repoUrl":args.repo_url,"classCategories":LABELS,
        "classCounts":[audit["splits"]["train"]["class_counts"][k] for k in LABELS],
        "lengthCategories":["1-40","41-80","81-120","121-160",">160"],
        "lengthCounts":[int(((lengths>a)&(lengths<=b)).sum()) for a,b in [(0,40),(40,80),(80,120),(120,160),(160,100000)]],
        "confusions":[{"name":NAMES[v],"counts":read_json(ART/"evaluation"/(selection[v]+".json"))["confusion_matrix"]} for v in NAMES]}
    (ROOT/"build/deck").mkdir(parents=True,exist_ok=True)
    (ROOT/"build/deck/presentation_data.json").write_text(json.dumps(content,indent=2),encoding="utf-8")
    print("Built report and defense slides from executed results")


if __name__=="__main__":
    main()
