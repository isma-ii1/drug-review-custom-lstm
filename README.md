# Drug Review Classification with Custom LSTM Cells

UIR Advanced Natural Language Processing, 2026/2027.

This project implements the course appendix's **Vanilla LSTM, CIFG, and WMC** cells manually in PyTorch with `nn.Linear`. It compares the same text classifier across a fixed configuration and an equal four-configuration validation search per architecture. No built-in `nn.LSTM` is used.

The target follows the supplied draft report: ratings 1–4 are Negative, 5–6 Neutral, and 7–10 Positive. These thresholds are a project design choice. **Only review text enters the classifier.** This is an NLP research project, not a medication recommendation system.

<!-- measured-results:start -->
## Measured results

All models use the same 48,257 unique held-out reviews. Selections use validation Macro-F1.

| Cell | Fixed accuracy | Fixed Macro-F1 | Selected setting | Selected accuracy | Selected Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| Vanilla | 65.80% | 0.5501 | batch_128 | 69.40% | 0.5583 |
| CIFG | 67.78% | 0.5484 | lr_0.001 | 70.50% | 0.5650 |
| WMC | 70.41% | 0.5583 | fixed | 70.41% | 0.5583 |

These are executed, reduced-data, single-seed results. They do not establish a universal architectural ranking.
<!-- measured-results:end -->

## Run locally

Python 3.11 or newer is required. A CPU is sufficient. Use a CUDA-compatible PyTorch installation for GPU experiments.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m drug_review.experiment prepare --train "path/to/drugsComTrain_raw.csv" --test "path/to/drugsComTest_raw.csv"
.venv\Scripts\python.exe -m drug_review.experiment train --threads 2
```

On Linux, macOS, Colab or Kaggle, replace `.venv\Scripts\python.exe` with the active environment's `python`. In Colab, upload both CSVs, clone this repository, then run the same `prepare` and `train` modules after `pip install -e ".[dev]"`.

The supplied source data is intentionally excluded from the public repository. Obtain the source files through your course's authorized dataset access and pass their local paths. Dataset reference specified by the course: [UCI dataset 462](https://archive.ics.uci.edu/dataset/462).

## Data protocol

1. Decode HTML entities, normalize case and whitespace, preserve negations and contractions.
2. Independently remove invalid reviews, texts with contradictory three-class labels, and repeated text within each source file.
3. Exclude every candidate training text found in the **raw** test source. This removes 32,111 cleaned training rows in the supplied files.
4. Stratify the eligible training pool 80/20 using seed 42. Select fixed stratified training and validation subsets of 15,000 and 3,000 reviews. Keep all 48,257 valid, unique, unambiguous test reviews.
5. Build a 12,000-entry vocabulary on the executed training subset only. Use the first 160 tokens and mask padded recurrent states.

`data/processed/data_audit.json` records the source hashes and exclusions. `split_manifest.csv` records original IDs, labels and normalized-text SHA-256 hashes. Each normalized text occurs in exactly one executed partition. We preserve class proportions rather than balancing the sample by discarding most Positive reviews. The model loss uses training-derived inverse-frequency class weights.

Scores describe this strict text-deduplicated population and are not directly comparable with scores on the unmodified raw split. Exact deduplication does not address all paraphrase or patient/group leakage.

## Models and experiments

Shared dimensions: embedding 64, hidden state 32, dropout 0.2, three output logits, seed 42, gradient norm clipping 1, class-weighted cross-entropy, maximum 6 epochs, early-stop patience 3.

| Setting | Optimizer | Learning rate | Batch size | Weight decay |
| --- | --- | --- | --- | --- |
| fixed | AdamW | 0.003 | 256 | 0.01 |
| lr_0.001 | AdamW | 0.001 | 256 | 0.01 |
| batch_128 | AdamW | 0.003 | 128 | 0.01 |
| adam | Adam | 0.003 | 256 | 0.01 |

Each setting runs for each cell: 12 experiments total. The Adam comparison includes its different handling of regularization. Batch-size changes alter the number of optimizer updates within an equal epoch budget. All runs use the same splits and primary metric, Macro-F1. Architecture winners and the overall candidate follow validation Macro-F1. The script saves the selections before scoring test data.

Input gate blocks use Xavier initialization, recurrent gate blocks use orthogonal initialization, and forget biases start at 1. WMC uses full learned memory projection matrices. Its output gate reads the **updated** memory, as required by the appendix. Parameter counts include embeddings and the head, and separately expose recurrent parameters.

The manual recurrent loop is scripted with TorchScript for CPU execution. Recent PyTorch versions emit a deprecation warning for TorchScript; the tested release still supports it. See `requirements-lock.txt` for the executed environment. GPU results may differ despite deterministic settings.

## Inspect results

- `artifacts/validation_results.csv`: all runs, selected epochs, parameters and elapsed times.
- `artifacts/runs/<run_id>/history.csv`: epoch-level training loss, validation accuracy/Macro-F1 and timing.
- `artifacts/test_results.csv`: fixed-setting and validation-selected test results.
- `artifacts/evaluation/`: per-class metrics, confusion matrices and aligned predictions.
- `artifacts/selection_before_test.json`: frozen validation selections.
- `artifacts/error_slices.csv`: exploratory error summaries.
- `artifacts/checkpoints/`: local trained weights, excluded from Git.
- `output/Drug_Review_Final_Report.pdf`: five-page report.
- `output/Drug_Review_Defense_Slides.pdf`: ten-slide English defense.
- `output/Drug_Review_Defense.pptx`: editable PowerPoint deck with native tables, charts and speaker notes.
- `output/Drug_Review_Defense.html`: local browser presentation with arrow-key navigation and N for speaker notes.
- `output/Speaker_Notes.md`: a 600-second speaking plan.

The HTML presentation and its adjacent `figures` directory must remain together. The PDF slides are self-contained. The PowerPoint deck uses the bundled presentation authoring runtime. Its PDF export had a font dependency error, so the matching static PDF uses high-resolution native slide renders while the PowerPoint retains editable charts and tables.

Predict from a saved checkpoint:

```powershell
.venv\Scripts\python.exe -m drug_review.predict --checkpoint artifacts/checkpoints/vanilla_fixed.pt --text "It helped, but the side effects were difficult."
```

Softmax scores are uncalibrated model outputs. Use the vocabulary and maximum length from the same experiment.

## Reproduce the report and slides

```powershell
.venv\Scripts\python.exe scripts/build_deliverables.py --repo-url https://github.com/isma-ii1/drug-review-custom-lstm
.venv\Scripts\python.exe scripts/verify_results.py
```

This requires all 12 completed runs and their test evaluations. Report generation reads the saved results and does not invent or hardcode performance scores. The written methods target the supplied default study and need revision if its data sizes or dimensions change. PowerPoint and matching slide-PDF regeneration additionally use `scripts/build_presentation.mjs` and the bundled Codex presentation runtime. Copy that builder into `build/deck/`, link its `node_modules` to the runtime's dependencies, and set `PRESENTATION_SKILL_DIR`, `RUNTIME_PYTHON`, and `RUNTIME_NODE_MODULES`. Use `FINAL_PPTX` to select a fresh output filename when regenerating. Training and report-PDF/HTML generation do not require that presentation runtime.

After the native slide renders are available, run `python scripts/render_slides_pdf.py` to create the matching PDF. The PowerPoint file contains short presenter prompts; use the slide timings and `output/Speaker_Notes.md` to rehearse the ten-minute explanation of the figures and equations.

To use the full eligible training and validation pools, prepare a **new** processed directory and use a **new** experiment output directory:

```powershell
.venv\Scripts\python.exe -m drug_review.experiment prepare --train "path/to/drugsComTrain_raw.csv" --test "path/to/drugsComTest_raw.csv" --train-limit 0 --val-limit 0 --output data/full
.venv\Scripts\python.exe -m drug_review.experiment train --data data/full --output artifacts/full --device auto
```

Do not choose new settings after inspecting the held-out test results. Further tuning requires a new planned validation study and honest disclosure of any repeated test use.

## Course requirements and remaining submission details

The implementation, architecture comparison, hyperparameter study, parameter counts, short report and English slides address the technical requirements in `Guidelines_RNN_2627.pdf`.

The guidelines also require a team of **5 or 6 students from the same lab group**. The supplied report names only Ismail Ighmour. Confirm the complete team roster and lab group before submitting. No additional identities have been invented.

The document states a submission deadline of **25 October 2026**, including that day, and a defense during lab 6. That stated deadline precedes the 1 November 2026 work date. Confirm any extension or updated submission instructions with the course. No submission form has been sent.

## References

- Younes Moukafih, *Lab Project on Recurrent Neural Networks*, Advanced NLP, 2026/2027. Supplied course guidelines and appendix.
- Supplied `drugsComTrain_raw.csv` and `drugsComTest_raw.csv`; exact hashes are recorded in the data audit.
- Greff et al., [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069).
- Landi et al., [Working Memory Connections for LSTM](https://arxiv.org/abs/2109.00020).

The appendix's equations control the implementation where a research paper studies a broader model family.
