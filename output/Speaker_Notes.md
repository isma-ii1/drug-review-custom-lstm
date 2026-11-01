# Defense speaker notes

Total planned time: 10 minutes (600 seconds).

Open Drug_Review_Defense.html and use Left/Right to navigate, N for notes, F for fullscreen.

## Slide 1: Drug review classification (35 seconds)

Our task is to infer rating-derived satisfaction from patient-written review text. We compare the standard LSTM gate design with two appendix variants. The purpose is to measure the effect of the memory mechanism under one common experimental protocol.

## Slide 2: Task and labels (50 seconds)

The supplied dataset contains a numerical satisfaction rating and a review. We follow the draft report's explicit three-class mapping. The guidelines require classification but do not prescribe these thresholds. The labels are a proxy for satisfaction, and a mixed review may not align neatly with its numerical rating. We exclude metadata and rating from inputs to avoid direct target leakage.

## Slide 3: Class imbalance and review lengths (60 seconds)

The raw files have 161,297 training rows and 53,766 test rows. The executed training subset is strongly imbalanced. This is why accuracy alone is insufficient and Macro-F1 is our primary measure. Review lengths support a 160-token limit: fewer than one percent of reviews exceed it. We use class-weighted loss with weights derived only from training data.

## Slide 4: Duplicate leakage and shared partitions (60 seconds)

Deduplication is one of the most consequential findings. A random split without checking text overlap could reward memorization. We normalize HTML, case and whitespace, remove within-file conflicting labels, and retain unique texts. We give the supplied test source priority and exclude every matching training text. We stratify the eligible training pool, then use fixed subsets permitted by the guidelines. The test set remains all cleaned unique eligible test reviews. This protocol changes the evaluation population, which limits comparison to raw-split published scores.

## Slide 5: Three custom recurrent cells (85 seconds)

All models share a 64-dimensional embedding, a 32-dimensional hidden state and the same classification head. Vanilla computes independent input, forget and output gates and a candidate. CIFG ties the input gate to the complement of the forget gate, reducing the cell's parameters. WMC adds learned full-matrix nonlinear projections of the previous memory into the input and forget gates. Its output gate uses the updated current memory. We implement every cell manually with nn.Linear and elementwise operations. We never use nn.LSTM. Masking preserves the final valid hidden state rather than a state produced by padding. Direct equation, gradient and padding checks passed.

## Slide 6: Parameter cost (60 seconds)

Recurrent and total parameter counts answer different questions. CIFG removes one set of gate transformations, which saves exactly a quarter of the recurrent parameters at the same dimensions. WMC adds three full hidden-to-hidden memory matrices. Because the embedding dominates this small classifier, cell savings translate into a much smaller percentage reduction for the complete model. This is why the report includes both counts and actual training times.

## Slide 7: Hyperparameter study (65 seconds)

Each architecture receives the same four configurations. The fixed comparison changes only the recurrent cell. The additional settings each change one factor, so their effects remain interpretable. We keep dropout at 0.2 and weight decay at 0.01. Adam and AdamW treat weight decay differently, so that is part of the optimizer comparison. All runs use seed 42, norm clipping at one and identical class weights. We select the checkpoint by validation Macro-F1 and freeze the configuration selections before evaluating test predictions.

## Slide 8: Held-out performance (80 seconds)

This table separates the fixed-setting architectural comparison from each architecture's validation-selected result. Macro-F1 gives every class the same weight. The report also includes precision and recall by class and full confusion matrices, since a model can score high accuracy by favoring Positive. The differences here describe a compact model trained on a reduced subset with one seed. They do not establish a universally superior cell. Test performance does not determine our chosen model.

## Slide 9: Errors and limits (60 seconds)

The confusion matrices show which true classes the selected models confuse. Neutral is a small class and its rating definition can cover mixed language. The project saves deterministic error examples and exploratory negation and length slices. Those subsets do not prove causal explanations for errors. We use one seed and a limited search, so small gaps may reflect optimization variation. Exact text deduplication cannot eliminate paraphrases or shared patient identity, and random splitting cannot establish temporal generalization.

## Slide 10: Conclusion and next experiments (45 seconds)

Our main contribution is a reproducible comparison that follows the prescribed cell equations and prevents exact text leakage across partitions. The overall candidate follows validation Macro-F1. The strongest next experiment is to retain the frozen protocol and increase training coverage with repeated seeds. A broader validation search and pretrained embeddings may improve quality. The complete PyTorch source, experiment logs, parameter counts and report are available at https://github.com/isma-ii1/drug-review-custom-lstm.

## Sources

Course guidelines and appendix, supplied training/test CSVs, artifacts/validation_results.csv and artifacts/test_results.csv.
Greff et al.: https://arxiv.org/abs/1503.04069
Landi et al.: https://arxiv.org/abs/2109.00020
https://github.com/isma-ii1/drug-review-custom-lstm