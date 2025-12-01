## Project overview

- Purpose: image classification of four mosquito species using EfficientNet-B2 models (training, evaluation, inference).
- Top-level dirs: `src/` (code), `data/csv/` (train/val/test CSVs with `filepath` and `species`), `experiments/` (models + predictions), `test_mosquito_img_samples/` (sample images).

## Big-picture architecture

- Training: `src/train/train_model1.py` (timm, LabelEncoder) and `src/train/train_model2.py` (torchvision EfficientNet weights, pandas category encoding). Both read CSVs at `data/csv/*.csv` and save state dicts with `torch.save`.
- Inference: `src/predict/predict.py` (single-image CLI), `src/predict/batch_test.py` (folder-based batch inference), `src/predict/confusion_matrix.py` (plotting using a CSV of predictions).
- Models are saved as state_dict files (e.g. `experiments/model1_baseline/*.pth`). Loading requires constructing the same model architecture then `load_state_dict`.

## Important conventions & patterns (explicit)

- CSV format: expected columns include `filepath` and `species`. Validation/prediction CSVs used by evaluation scripts expect at least `true_label` and `pred_label` columns.
- Label encoding: training scripts differ — `train_model1.py` uses `sklearn.preprocessing.LabelEncoder` (stores mapping in `encoder` at runtime), while `train_model2.py` uses `pandas` categorical codes. When loading models for inference, ensure class ordering matches training.
- Paths: several scripts use absolute OneDrive paths (e.g. in `src/predict/*.py` and `src/predict/confusion_matrix.py`). Agents should convert those to repo-relative paths or expose them as config variables when modifying code.
- Mixed precision: `train_model2.py` uses `torch.cuda.amp.GradScaler()` and autocast; `train_model1.py` does not. Preserve AMP behavior when fine-tuning or reusing training loops.

## Developer workflows (commands & examples)

- Train model1 (baseline):
  - `python src/train/train_model1.py`
- Train model2 (augmented + partial-unfreeze):
  - `python src/train/train_model2.py`
- Predict single image:
  - `python src/predict/predict.py` and paste the full image path when prompted, or call `predict(image_path, weights_path)` from a small wrapper.
- Batch test (folder of class subfolders):
  - `python src/predict/batch_test.py` (edit `MODEL_PATH`/`TEST_FOLDER` at top or refactor to accept CLI args).
- Generate confusion matrix from predictions CSV:
  - `python src/predict/confusion_matrix.py` (edit `csv_path` variable to point at `experiments/predictions/val_predictions.csv`).

## Integration points & dependencies

- Python libs: `torch`, `torchvision`, `timm`, `pandas`, `PIL`, `sklearn`, `matplotlib`, `seaborn`. Use the same major PyTorch version (scripts assume modern torch + torchvision and timm compatibility).
- Saved artifacts: `experiments/` stores trained `.pth` state_dicts and `experiments/predictions/` stores CSVs used by plotting scripts.

## Guidance for AI agents editing this repo

- Prefer repo-relative paths. Replace hard-coded absolute OneDrive paths with configurable constants or CLI args.
- When changing model save/load, ensure `state_dict` semantics are preserved: scripts save `model.state_dict()` and loaders call `model.load_state_dict(torch.load(...))`.
- When adding CLI/automation, make each script accept `--weights`/`--csv`/`--out` args instead of editing top-level vars.
- Preserve existing label ordering: search for `LabelEncoder` or `.astype('category').cat.codes` to find where encodings originate; add a small `classes.json` next to `.pth` when adding reproducible loading.

## Quick file references (examples)

- `src/train/train_model1.py` — basic pipeline using `timm`, `LabelEncoder`, saves `model1_efficientnet_b2_best.pth`.
- `src/train/train_model2.py` — augmented transforms, partial unfreeze, AMP, saves under `experiments/model2_augmented/`.
- `src/predict/predict.py` / `src/predict/batch_test.py` — inference patterns; update these to use relative paths or CLI flags.

If anything above is unclear or you want additional examples (e.g., a CLI refactor or a helper to persist `classes.json`), tell me which area to expand.
