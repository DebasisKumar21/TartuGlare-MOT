# Glare-Robust Pedestrian Detection — Training Pipeline

Colab notebooks to train and evaluate three YOLO11s variants for pedestrian detection under solar glare, and reproduce the paper's headline SolarDrive results.

## What this builds

| Variant | Description |
|---|---|
| A | Clean MOT17 baseline, no augmentation |
| B | MOT17 + generic Albumentations brightness/contrast/gamma jitter (fair-comparison baseline) |
| C | MOT17 + physics-calibrated glare pipeline: highlight saturation, shadow crush, stroboscopic flicker, spatial masking, sampled across 5 discrete RIL severity levels (0/10/20/30/36%), calibrated against the SolarDrive dataset |

## Requirements

- A Google Drive with the [MOT17](https://motchallenge.net/data/MOT17/) `train/` split uploaded
- Google Colab (T4 GPU runtime; a free-tier account is enough, sessions are pre-split to fit)
- For the real-SolarDrive evaluation step: access to the SolarDrive test sequences (not included in this repo — see the SolarDrive paper for access)

## Run order

1. **`01_setup_and_data.ipynb`** — edit `MOT17_TRAIN` and `DRIVE_ROOT` at the top, then run top to bottom. Builds all three datasets plus the severity-curve evaluation set. Long-running (mostly CPU/IO); safe to reconnect and re-run, already-built files are skipped.
2. **`02a` / `02b`** — train Variant A (Part 1, then Part 2; ~2.5h each)
3. **`03a` / `03b`** — train Variant B
4. **`04a` / `04b`** — train Variant C
5. **`05_evaluation_and_export.ipynb`** — edit `SOLAR_TEST` to your SolarDrive test-sequence path, then run top to bottom. Produces Table 1, the SolarDrive comparison figure, the severity-curve headline figure, and a final paper-asset checklist.

Each training notebook can be run independently once `01` has finished — there's no requirement to train variants in order, or on the same machine/session.

## A note on calibration

SolarDrive's Table II reports measured highlight/shadow saturation fractions only at peak glare severity (~36% RIL). The intermediate 10/20/30% targets used to build Variant C, and the assignment of measured stroboscopic-flicker frequencies to MOT17 sequences (which have no glare of their own), are both interpolated/synthetic design choices rather than direct field measurements. Both are called out explicitly, with the reasoning, at the cell where they're defined in `01_setup_and_data.ipynb` — look for the comments before treating Variant C numbers as final.

## Outputs

Everything is written to `DRIVE_ROOT` (default `/content/drive/MyDrive/glare_detection`): the three built datasets, trained weights per variant, all figures, and the final tables/summary text used directly in the paper.

## Citation

If you use this pipeline, please cite the SolarDrive dataset paper (citation details forthcoming).
