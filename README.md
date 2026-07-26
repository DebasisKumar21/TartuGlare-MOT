# Glare-Robust Pedestrian Detection — Training Pipeline

Colab notebooks that train and evaluate three YOLO11 variants for pedestrian detection under solar glare, and reproduce the paper's SolarDrive results. This README assumes no prior familiarity with MOT17 or this codebase.

## What this pipeline builds

Three training sets, three trained models, evaluated against each other:

| Variant | What it is |
|---|---|
| **A** | Clean MOT17, no augmentation (baseline) |
| **B** | MOT17 + generic Albumentations brightness/contrast/gamma jitter (fair-comparison baseline — *not* physics-calibrated) |
| **C** | MOT17 + physics-calibrated glare pipeline: highlight saturation, shadow crush, stroboscopic flicker, spatial masking, sampled across 5 RIL severity levels (0/10/20/30/36%) calibrated against the SolarDrive dataset |

The final notebook evaluates all three on real SolarDrive footage and on a synthetic severity sweep, producing the paper's tables and figures.

---

## Step 0: What you need before starting

1. A Google account with Google Drive and Google Colab access (the free tier works — see runtime notes below).
2. The **MOT17 dataset** (see Step 1 — the official download is currently unreliable, so this uses a Kaggle mirror instead).
3. A free **Kaggle account and API token**, only if you use the automated download in Step 1.
4. Access to the **SolarDrive test sequences** (only needed for the final evaluation notebook, not for setup or training — see Step 2).

---

## Step 1: Get MOT17

**The official download at motchallenge.net is currently unreliable/down.** Two options:

- **Recommended — let the notebook do it.** `01_setup_and_data.ipynb` Step 2b downloads a community-mirrored copy of MOT17 from Kaggle ([wenhoujinjust/mot-17](https://www.kaggle.com/datasets/wenhoujinjust/mot-17)) directly into Colab and copies it onto your Drive. You'll need a free Kaggle account and API token (Kaggle → Settings → API → "Create New Token" — downloads a `kaggle.json` file); the notebook prompts you to upload it when needed. This only runs once — later sessions detect MOT17 already on Drive and skip it automatically. **If you're using this route, skip straight to Step 3 below** — the notebook handles download, extraction, and a structural sanity-check together.
- **Manual — if you already have MOT17 from elsewhere**, or motchallenge.net comes back online. Download `MOT17.zip` (or `MOT17Labels.zip` + the images — you need both images and ground-truth annotations), unzip it locally, and continue below.

Either way, you need this exact structure:

   ```
   MOT17/
   └── train/
       ├── MOT17-02-DPM/
       │   ├── img1/            <- .jpg frames
       │   ├── gt/gt.txt        <- ground-truth annotations
       │   └── seqinfo.ini      <- frame rate, resolution, etc.
       ├── MOT17-04-DPM/
       ├── MOT17-05-DPM/
       └── ... (more sequences)
   ```

   **This exact `train/<sequence>/img1/`, `train/<sequence>/gt/gt.txt`, `train/<sequence>/seqinfo.ini` structure is required.** MOT17 ships three detector variants per scene (`MOT17-02-DPM`, `MOT17-02-FRCNN`, `MOT17-02-SDP`, etc.) — that's expected, not a duplicate; the pipeline treats each folder as an independent sequence and you don't need to merge or deduplicate them.

   Only the `train/` split is used — it's the only one with ground-truth labels (MOT17's `test/` split has no `gt.txt`, so it can't be used for supervised training or validation here).

   **If you download manually**, upload the whole `MOT17/` folder (with its `train/` subfolder intact) to your Drive, e.g. `MyDrive/glare_detection_data/MOT17/train/...`. For large uploads, Google Drive for Desktop or a zip-then-unzip-in-Colab approach is usually faster than dragging thousands of files into the browser.

## Step 2: Get access to SolarDrive test sequences (only needed later)

You don't need this yet — it's only required by the final notebook (`05_evaluation_and_export.ipynb`). Get the SolarDrive test sequences from wherever you're storing/sharing that dataset, and upload them to Drive the same way, e.g.:

```
MyDrive/SolarDrive_test/sun_glare_0/
MyDrive/SolarDrive_test/sun_glare_1/
MyDrive/SolarDrive_test/sun_glare_2/
MyDrive/SolarDrive_test/sun_glare_3/
```

## Step 3: Open the first notebook and set two paths

1. Open `01_setup_and_data.ipynb` in Google Colab.
2. **Runtime > Change runtime type > T4 GPU**, then Save.
3. Run cells top to bottom. **Step 2** of the notebook ("Get MOT17, Mount Drive, and Set Paths") has three sub-parts:
   - **2a** mounts your Drive.
   - **2b** downloads MOT17 from Kaggle — skip this cell if you already have MOT17 on Drive (manual route from Step 1).
   - **2c** is where you edit two lines:

     ```python
     MOT17_TRAIN = '/content/drive/MyDrive/MOT17/train'
     DRIVE_ROOT  = '/content/drive/MyDrive/glare_detection'
     ```

     `MOT17_TRAIN` → the path confirmed by 2b, or wherever you uploaded MOT17 yourself (must point *at* the `train` folder itself, not its parent). `DRIVE_ROOT` → any folder for all pipeline outputs; created automatically if it doesn't exist.
4. Continue running the rest of the notebook top to bottom. It prints `MOT17 sequences found: <N>` early on — if that's `0`, the `MOT17_TRAIN` path is wrong; double check it against Drive before continuing.

---

## Step 4: Run everything in order

Run these Colab notebooks **in this exact order**. Each one states at the top what it needs before it can run. No further splitting is needed beyond this — each training notebook is sized to fit a single free-tier Colab session.

| # | Notebook | What it does | Approx. runtime |
|---|---|---|---|
| 1 | `01_setup_and_data.ipynb` | Downloads MOT17 (if needed) and builds all 3 training datasets + the severity evaluation set | Long (CPU/IO-bound; varies with download/Drive speed) |
| 2 | `02a_train_variantA_part1.ipynb` | Train Variant A, epochs 1–25 | ~2.5h |
| 3 | `02b_train_variantA_part2.ipynb` | Train Variant A, epochs 26–50 | ~2.5h |
| 4 | `03a_train_variantB_part1.ipynb` | Train Variant B, epochs 1–25 | ~2.5h |
| 5 | `03b_train_variantB_part2.ipynb` | Train Variant B, epochs 26–50 | ~2.5h |
| 6 | `04a_train_variantC_part1.ipynb` | Train Variant C, epochs 1–25 | ~2.5h |
| 7 | `04b_train_variantC_part2.ipynb` | Train Variant C, epochs 26–50 | ~2.5h |
| 8 | `05_evaluation_and_export.ipynb` | Evaluates all 3 variants, produces tables + figures | Short–medium |

**Practical notes:**
- Notebooks 2–4 (training) can run in **any order relative to each other** — they're independent once `01` has finished. Within a variant, Part 1 must finish before Part 2 (Part 2 checks for this automatically and stops with a clear error if it hasn't).
- **Part 2 is not a literal continuation of Part 1's training schedule** — it's a fresh 25-epoch run warm-started from Part 1's best checkpoint, with its own optimizer/LR schedule, not `resume=True` carried across both parts. This is a legitimate staged fine-tuning approach, but if you're writing up the training protocol for the paper, state it exactly that way (each Part 2 notebook has the precise wording to use) rather than describing it as "one continuous 50-epoch run" — a reviewer familiar with Ultralytics may otherwise ask why `resume=True` wasn't used end-to-end. The exact schedule is also saved into `training_summary.json` for traceability.
- Every notebook is safe to **stop and re-run from the top** at any point — completed work is detected and skipped, not redone. This matters because free-tier Colab disconnects idle sessions; just reopen and re-run all cells.
- Training is split into two parts per variant specifically because a full 50-epoch run (3–5h) is longer than a free Colab session reliably survives in one sitting.

## Step 5: Where to find your results

Everything lands in the `DRIVE_ROOT` folder you set in Step 3 (default: `/content/drive/MyDrive/glare_detection`):

```
glare_detection/
├── yolo_dataset_A_clean/       <- Variant A images + labels
├── yolo_dataset_B_standard_aug/
├── yolo_dataset_C_ril_calibrated/
├── severity_eval/              <- fixed 5-level synthetic eval set
├── results/                    <- training run checkpoints
├── figures/                    <- fig1, fig2, fig3 (paper-ready PNGs)
├── table1_for_paper.md         <- paste directly into the manuscript
├── table1_for_paper.csv
├── results_summary.txt         <- includes a suggested abstract sentence
└── variant_{A,B,C}_weights_path.txt
```

The last cell of `05_evaluation_and_export.ipynb` prints a checklist confirming every one of these exists before you start writing.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `MOT17 sequences found: 0` | `MOT17_TRAIN` path is wrong, or points above/below the actual `train/` folder — check the exact path in Drive. |
| Kaggle download cell asks for `kaggle.json` every session | The token wasn't saved to a persistent location — expected on Colab's local disk (it resets each session), but this only matters if Step 2b runs again, which it won't once MOT17 is confirmed on Drive. |
| `WARNING: no GT for <sequence>` | That sequence folder is missing `gt/gt.txt` — re-check your MOT17 source included ground truth, not just images. |
| Notebook raises `RuntimeError: Part 1 has not finished` | You're running a Part 2 notebook before its Part 1 finished. Run and complete Part 1 first. |
| `nb1_paths_config.json not found` | You haven't run `01_setup_and_data.ipynb` to completion yet (or on a different `DRIVE_ROOT` than this notebook is looking for). |
| Training seems to restart from epoch 0 after a disconnect | Shouldn't happen — checkpoints save every 5 epochs. If it does, check that `DRIVE_ROOT`/`RESULTS_DIR` weren't changed between runs. |
| `SolarDrive test images not found` in notebook 05 | Expected until you complete Step 2 above and update `SOLAR_TEST` in that notebook's Step 3 cell. |

---

## A note on calibration

SolarDrive's Table II reports measured highlight/shadow saturation fractions only at peak glare severity (~36% RIL). The intermediate 10/20/30% targets used to build Variant C, and the assignment of measured stroboscopic-flicker frequencies to MOT17 sequences (which have no glare of their own), are interpolated/synthetic design choices rather than direct field measurements. Both are called out explicitly, with reasoning, in `01_setup_and_data.ipynb` Step 3 (Augmentation Module Library) — check the comments there before treating Variant C numbers as final.

## Citation

If you use this pipeline, please cite the SolarDrive dataset paper (citation details forthcoming).
