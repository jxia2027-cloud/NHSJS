# Review Packet for NHSJS CKD Manuscript

This packet is meant to be pasted or uploaded to an AI reviewer that cannot accept raw Python or JSON files. It does not include full source-code or raw JSON payloads. Instead, it gives the reviewer the manuscript/PDF locations, repository link, run configuration, and key output tables needed to audit the paper.

## Files To Review

- Rendered PDF: `paper/build/manuscript.pdf`
- Root PDF copy: `When_Perfect_Is_Wrong.pdf`
- Review-copy Word document: `When_Perfect_Is_Wrong_Review_Copy.docx`
- LaTeX source: `paper/manuscript.tex`
- Review decision PDF: `Justin - Review Decision (1).pdf`
- Submission guidelines PDF: `NHSJS - Submission Guidelines.pdf`
- Analysis script path: `datasetsPythonScripts/nhsjs.py`
- Output tables path: `outputs/latex_data/`
- Full result summary path: `outputs/results.json`
- GitHub repository: https://github.com/jxia2027-cloud/NHSJS.git

## Current Build

- Compile command: `cd paper && /Users/jxchu/bin/tectonic manuscript.tex --outdir build`
- Current rendered page count: 18 pages
- Abstract length: about 221 words
- Citation audit: 29 bibliography entries, 29 cited, no missing citation keys, no unused references
- Tables intentionally use smaller font sizes for readability and layout.

## Analysis Run Configuration

- Learning-curve repeats: `N_REPS=20`
- NHANES repeats: `N_REPS_NH=20`
- Tawam repeats: `N_REPS_TW=20`
- Power-law bootstrap iterations: `N_BOOT=500`
- Splits are stratified by label.
- Missing values and scaling are fit on training data only.
- UCI nominal variables are one-hot encoded.
- Clinically ordered UCI variables keep explicit ordered encodings.

## Reviewer-Requested Experiments Now Present

- UCI-to-external transfer using shared clinical features.
- NHANES feature sensitivity with three feature sets:
  - all features
  - no eGFR/ACR
  - no eGFR/ACR/Cr/BUN
- 20-repeat stratified learning curves.
- UCI single-feature logistic screening.
- UCI top-feature ablation.
- Tawam train/test gap monitoring.
- Calibration/threshold summaries via Brier score and sensitivity/specificity.
- KDIGO stage mapping table.
- Power-law fit table with estimates, bootstrap 95% CIs, R2, and drop-2 sensitivity ceiling.
- In-text citations added for dataset provenance, CKD burden, KDIGO staging, NHANES survey guidance, shortcut learning, external validation, small-sample validation risk, and reporting guidance.

## Key Output Tables

### Held-Out Full-Data Results

| Model | UCI AUC | NHANES AUC | Tawam AUC | NHANES Brier | Tawam Brier |
|---|---:|---:|---:|---:|---:|
| LogReg | 1.0000 | 0.9457 | 0.8768 | 0.0835 | 0.0612 |
| RF | 0.9996 | 1.0000 | 0.9083 | 0.0043 | 0.0658 |
| HGB | 1.0000 | 1.0000 | 0.9102 | 0.0001 | 0.0576 |
| MLP | 0.9983 | 0.9971 | 0.9083 | 0.0153 | 0.1397 |
| Ensemble | 1.0000 | 1.0000 | 0.9050 | 0.0033 | 0.0663 |

### NHANES Feature-Set Sensitivity

| Model | All features | No eGFR/ACR | No eGFR/ACR/Cr/BUN |
|---|---:|---:|---:|
| LogReg | 0.9457 | 0.8131 | 0.6926 |
| RF | 1.0000 | 0.9737 | 0.9218 |
| HGB | 1.0000 | 0.9751 | 0.9205 |
| MLP | 0.9971 | 0.9690 | 0.9155 |
| Ensemble | 1.0000 | 0.9749 | 0.9216 |

### UCI-Trained Cross-Dataset Transfer

| Model | UCI to UCI common features | UCI to NHANES | UCI to Tawam |
|---|---:|---:|---:|
| LogReg | 0.9194 | 0.3591 | 0.5267 |
| RF | 0.9758 | 0.7670 | 0.6163 |
| HGB | 0.9663 | 0.4223 | 0.6414 |
| MLP | 0.9605 | 0.3284 | 0.7695 |
| Ensemble | 0.9767 | 0.4461 | 0.6278 |

### Threshold 0.5 Sensitivity/Specificity

| Model | Dataset | Sensitivity | Specificity |
|---|---|---:|---:|
| LogReg | NHANES | 0.963 | 0.742 |
| LogReg | Tawam | 0.357 | 1.000 |
| RF | NHANES | 1.000 | 1.000 |
| RF | Tawam | 0.143 | 1.000 |
| HGB | NHANES | 1.000 | 1.000 |
| HGB | Tawam | 0.429 | 0.982 |
| MLP | NHANES | 0.989 | 0.961 |
| MLP | Tawam | 0.786 | 0.917 |
| Ensemble | NHANES | 1.000 | 1.000 |
| Ensemble | Tawam | 0.286 | 1.000 |

## Specific Review Questions

Please review the manuscript for:

1. Whether every reviewer comment is fully addressed.
2. Whether the paper follows NHSJS formatting and content guidelines.
3. Whether the in-text citations actually support the claims attached to them.
4. Whether any references are inaccurate, weak, outdated, missing, or unnecessary.
5. Whether the statistical claims match the output tables above.
6. Whether the methods are reproducible and clear enough.
7. Whether figures/tables are readable and not overlapping.
8. Whether conclusions are too strong for the evidence.
9. Whether limitations should be expanded.
10. Whether the GitHub/code/data availability statement is appropriate.

## Requested Review Format

Return findings in this structure:

- Major issues that must be fixed before submission
- Minor issues or polish suggestions
- Reviewer comments still partially addressed
- Citation/source problems
- Figure/table formatting problems
- Statistical/methodological inconsistencies
- Final recommendation: ready / almost ready / not ready

Be specific. Cite exact page numbers, section names, table/figure numbers, and LaTeX line numbers where possible.
