# SpikeNet2: IED detection with hard-negative mining (CV5)

Training and evaluation of SpikeNet2 for interictal epileptiform discharge (IED) detection on Henry Ford scalp EEG, using patient-level 5-fold cross-validation and hard-negative mining on IED-free control EEG.

## Data

| | Patients | Content |
|---|---|---|
| IED patients | 10 | 219 IEDs, 1,314 labeled 2 s clips (219 IED / 1,095 background) |
| Control (no IEDs) | 3 | 2 for mining (~2.2 h), 1 held out for FP/hour (~1 h) |

- Raw: 50 channels, 200 Hz (`CrossSubject_OUTPUT/<patient>/RunN_X.npy` + `RunN_meta.json`; controls in `no_ied/`).
- Clips: 2 s, resampled to 128 Hz, saved in `hf_spikenet/Events/` (controls as `ctrl_*.npy`).
- Model input: 37-channel bipolar montage, central 1 s (37 × 128).

## Pipeline

```
create_cv_data.ipynb         → hf_spikenet/CV5/lut_all_fold{0-4}.csv   (one fold assignment per file)
train_hard_from_scratch.ipynb
  for each fold:
    round 0      train on labeled clips
    round 1..R   mine control windows → add as negatives (H) → retrain a fresh model
    evaluate     best round on val → threshold on val (Youden) → test once
  → mean ± std over 5 folds
```

Each fold: 2 test patients, 2 val patients, 6 train patients; every patient is tested once. Controls are never in train/val/test.

## Key settings (`train_hard_from_scratch.ipynb`)

| Variable | Meaning |
|---|---|
| `METHOD` | run name; prefixes all outputs (`sanity`, `thr`, `topk`, …) |
| `TAUS` | mining rule per round: **float** = threshold (paper, e.g. `[0.8, 0.7, 0.6, 0.5]`), **int** = top-K (e.g. `[100, 200, 400]`) |
| `beta` (in `run_fold`) | class-balanced loss β (Cui et al., 2019), recomputed each round |
| `config` (`sleeplib/config.py`) | LR 1e-4, batch 256, max epochs, early-stopping patience |

## Run

1. `create_cv_data.ipynb`: writes the 5 fold luts (once).
2. `train_hard_from_scratch.ipynb`, top to bottom:
   - sanity cell: fold 0, one mining round;
   - CV5 cell: all folds. Set `METHOD` and `TAUS` first; run once per method.

## Outputs

`run/<timestamp>/` (checkpoints in `checkpoints/<timestamp>/`):

| File | Content |
|---|---|
| `log.txt` | per-epoch losses, class weights, val AUROC per round |
| `loss_{METHOD}_f{k}_r{r}.png` | train/val loss per fold and round |
| `{METHOD}_rounds.csv` | val AUROC and number of hard negatives per round |
| `{METHOD}_cv5_metrics.csv` | test metrics per fold, with mining (best round) |
| `{METHOD}_nomine_cv5_metrics.csv` | test metrics per fold, round 0 only (no mining) |
| `{METHOD}_test_roc_cv5.png` | test ROC of all folds plus the mean |

Metrics: AUROC, AUPRC, sensitivity, specificity, F1, FP/hour.

## Known issues

- About 23 of 37 montage channels are zero in the model input (montage or `extremes_remover`); this limits all current results.
- The paper's thresholds assume calibrated scores. With current scores (~0.2 on background), `τ = 0.8` mines almost nothing, which is why the top-K option exists.
- No patient data, clips or checkpoints are committed.