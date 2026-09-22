# Task A: current model vs AI-agent model

## Purpose
Post-hoc / exploratory comparison of the current Task A CXR model and the independently developed AI-agent Task A CXR model on the same fixed Test set.

## Data integrity checks
- Current model Test predictions: 128 patients
- AI-agent Test predictions: 128 patients
- Matched Subject IDs: 128/128
- Duplicate Subject IDs: none
- True labels matched for all patients
- Deaths: 17
- Survivors: 111

## Performance
| Model | ROC-AUC | PR-AUC | Brier score |
|---|---:|---:|---:|
| Current | 0.907260 | 0.622926 | 0.101732 |
| AI-agent | 0.834658 | 0.457562 | 0.105808 |

## Paired DeLong comparison
- ROC-AUC difference (current - AI-agent): 0.072602
- 95% CI: 0.007022 to 0.138182
- Two-sided unadjusted p-value: 0.030017

## Interpretation
This was not a prespecified primary hypothesis test and should be interpreted as an exploratory, post-hoc comparison.
The current model had a higher ROC-AUC point estimate than the AI-agent model on this Test set.
Because the Test set contained only 17 deaths, uncertainty remains substantial.
The two pipelines also differed in several design and training choices, so this comparison does not identify which individual design choice caused the performance difference.
