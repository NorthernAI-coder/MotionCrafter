# Preprocess Scripts 🧰

Welcome to the dataset prep room. ✨ The scripts in this folder turn raw datasets
into the MotionCrafter format that training code expects:

- `*_rgb.mp4`: video clips
- `*_data.hdf5`: geometry, camera, flow, mask, and point-map annotations
- `meta_infos.txt`: the index file that tells the loader what exists

Most datasets walk through the same front door: run one `gen_*.py` script, get
clips plus annotations, then normalize if needed. Kubric takes a small detour,
called out below.

## 0) The Fast Path 🚀

Use the unified launcher when the dataset has a normal one-step preprocess flow:

```bash
python run_preprocess.py \
  --dataset spring \
  --data-dir /path/to/raw/Spring \
  --output-dir /path/to/unnormed/Spring_video \
  --split train \
  --clip-length 150
```

Datasets with multiple splits can pass them as a comma-separated list:

```bash
python run_preprocess.py \
  --dataset gta_sfm \
  --data-dir /path/to/raw/GTA-SfM \
  --output-dir /path/to/unnormed/GTA-SfM_video \
  --splits train,test
```

## Kubric: The Two-Step Route 🧭

Kubric is the special case. 🌟 Before `gen_kubric_video.py` can make MotionCrafter
clips, it needs dense tracking files. Think of this as giving Kubric frames their
motion passport first.

1. Edit the paths and GPU ids at the top of `preprocess_kubric.sh`.
2. Generate dense tracking:

```bash
cd datasets/preprocess
./preprocess_kubric.sh
```

This creates processed Kubric frame folders, camera files, and
`*_dense_tracking_*.npy` files. If `SPLIT="validation"`, the tracking output goes
to `${PROCESSED_DIR}_val`.

3. Convert that processed tracking directory into normal MotionCrafter clips:

```bash
python run_preprocess.py \
  --dataset kubric \
  --data-dir /path/to/processed/kubric_val \
  --output-dir /path/to/unnormed/Kubric_video \
  --clip-length 18
```

After this, Kubric rejoins the regular pipeline. ✅

## 1) Shared Knobs 🎛️

All `gen_*.py` scripts understand the same core environment variables:

- `MOTIONCRAFTER_DATA_DIR`: raw or intermediate dataset root
- `MOTIONCRAFTER_OUTPUT_DIR`: output directory for converted videos and HDF5 files
- `MOTIONCRAFTER_CLIP_LENGTH`: frames per output clip

Some scripts also listen to dataset-specific switches:

- `MOTIONCRAFTER_SPLITS`: comma-separated split list, such as `train,test`
- `MOTIONCRAFTER_SPLIT`: one split name
- `MOTIONCRAFTER_PROCESS_SCENE_FLOW`: `true` or `false` for Point Odyssey

Example without the launcher:

```bash
cd datasets/preprocess
MOTIONCRAFTER_DATA_DIR=/path/to/raw/Spring \
MOTIONCRAFTER_OUTPUT_DIR=/path/to/unnormed/Spring_video \
MOTIONCRAFTER_SPLIT=train \
MOTIONCRAFTER_CLIP_LENGTH=150 \
python gen_spring_video.py
```

## 2) Normalize The Results 🧼

Once the unnormalized datasets are built, `normalize_video_dataset.py` resizes and
packs them into the normalized training layout:

```bash
python normalize_video_dataset.py \
  --data-dirs /path/to/unnormed/Spring_video /path/to/unnormed/GTA-SfM_video \
  --output-root /path/to/tmp_datasets \
  --resolution 320 640 \
  --num-workers 8 \
  --skip-existing
```

If `--output-root` and `--output-dirs` are omitted, output defaults to one of:

- the input path with `unnormed_datasets` replaced by `tmp_datasets`
- `data_dir/normalized` when that replacement is not possible

## 3) Build The Latent Index 🗂️

`preprocess_meta_file.py` creates the latent-side `meta_infos.txt` from generated
source meta files:

```bash
python preprocess_meta_file.py \
  --data-dirs /path/to/data_normed_1/Spring_video /path/to/data_normed_1/GTA-SfM_video \
  --latent-dirs /path/to/latent/Spring /path/to/latent/GTA-SfM
```

## 4) Field Notes 📝

- Most scripts expect CUDA for point maps or flow-heavy preprocessing.
- Keep clip length aligned with the training config you plan to use.
- Dataset folder layouts matter; each script expects its dataset's native shape.
- Shared helpers for env parsing and `meta_infos.txt` writing live in `preprocess_common.py`.

## 5) Dataset-Specific Dependencies 🧩

- Dynamic Replica requires `pytorch3d`.
- Virtual KITTI 2 requires `pandas`.
- EXR-reading scripts, such as IRS or Matrix City, require `OpenEXR` runtime support.

If `pip install -r requirements.txt` cannot install `pytorch3d` in your environment,
install it separately using the official PyTorch3D instructions that match your
PyTorch and CUDA versions.
