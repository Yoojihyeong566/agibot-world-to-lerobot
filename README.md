# agibot2lerobot

Convert [**AgiBot World 2026**](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)
downloads into ready-to-use **LeRobot v3.0** datasets, with one command.

AgiBot World ships datasets in **LeRobot v2.1** format, packed as `.tar.gz`
archives. Modern `lerobot` (>= 0.5) only reads **v3.0**, so every dataset must be
**extracted** and **version-converted (v2.1 → v3.0)** before you can train on it
or open it in `lerobot-dataset-viz`. This repo automates that on any machine.

> ⚠️ The AgiBot-provided `split_episodes_tool` does **not** work on these
> datasets — they contain no "Task Frame" annotations, so it outputs 0 episodes.
> We use lerobot's own `convert_dataset_v21_to_v30` instead.

---

## 1. Setup (conda)

```bash
git clone https://github.com/<you>/agibot-world-to-lerobot.git
cd agibot-world-to-lerobot

# Creates the `agibot_world` env, installs everything, applies the 2 fixes below.
bash scripts/setup_env.sh
conda activate agibot_world
```

`scripts/setup_env.sh` is idempotent and handles two **non-obvious** issues on
Linux (skip it only if you know what you're doing):

| Fix | Why |
|-----|-----|
| **ffmpeg = 7** (`environment.yml`) | lerobot decodes video with **torchcodec**, which supports ffmpeg **4–7 only**. conda may install ffmpeg 8 → `Could not load libtorchcodec`. |
| **`LD_LIBRARY_PATH` activate hook** | ffmpeg pulls `libopenvino`, which needs `CXXABI_1.3.15`. The system `libstdc++` is often too old → load conda's newer one first. The hook is written to `$CONDA_PREFIX/etc/conda/activate.d/`. |

Verify the env:
```bash
python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')"
```

<details>
<summary>Manual setup (no script)</summary>

```bash
conda env create -f environment.yml
conda activate agibot_world
pip install -e .
# fix: ensure ffmpeg 7
conda install -y -c conda-forge "ffmpeg=7.*"
# fix: prepend conda libstdc++ on activation
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"' \
  > "$CONDA_PREFIX/etc/conda/activate.d/zz_ld_library_path.sh"
```
`environment.full.yml` / `requirements.lock.txt` are the exact tested freezes
(reference only — prefer `environment.yml`).
</details>

---

## 2. Download AgiBot data (only what you need)

The full dataset is ~8.5 TB; download per task. Uses the HuggingFace CLI
(`pip install huggingface_hub`, then `hf auth login`).

```bash
# one ImitationLearning / RichInteraction archive (an episode range)
hf download agibot-world/AgiBotWorld2026 \
  RichInteraction/CommercialSpaces/task_4439/458713_458713.tar.gz \
  --repo-type dataset --local-dir ./AgiBotWorld2026

# a simulation "lite" set (data + meta + videos)
hf download agibot-world/AgiBotWorld2026 \
  --repo-type dataset --local-dir ./AgiBotWorld2026 \
  --include "simulation/take_cup_to_cart/g2_swift_picker/lite/*"
```

Keep the repo's directory structure (`--local-dir ./AgiBotWorld2026`) — the
converter discovers datasets from it automatically.

---

## 3. Convert → LeRobot v3.0

```bash
# see what will be converted
python -m agibot2lerobot list ./AgiBotWorld2026

# convert everything found, into ./AgibotWorld_lerobot/agibot/<NAME>/
python -m agibot2lerobot convert ./AgiBotWorld2026 ./AgibotWorld_lerobot --workers 8

# or just some, by name substring
python -m agibot2lerobot convert ./AgiBotWorld2026 ./AgibotWorld_lerobot --only RI_task_4439 SIM_take_cup
```

Output layout (each folder is a complete v3.0 dataset):
```
AgibotWorld_lerobot/agibot/
├── IL_task_4542_553049_553079/   {meta,data,videos}
├── RI_task_4439_458713_458713/
└── SIM_take_cup_to_cart/
```

Options: `--namespace` (default `agibot`), `--keep-extracted`,
`--keep-v21-backup`, `--workers N`.

**What it does per dataset:** extract the (multi-part) `.tar.gz` → locate the
v2.1 root (IL/RI nest an extra `data/`; simulation doesn't) → copy to output →
run `convert_dataset_v21_to_v30` in place → drop the `*_old` backup.

---

## 4. Visualize

`--root` is the **dataset folder itself** (not its parent), and `--repo-id` is
just a label. Print the exact command with:

```bash
python -m agibot2lerobot viz ./AgibotWorld_lerobot/agibot/RI_task_4439_458713_458713 --episode 0
```

which gives:

```bash
DS=./AgibotWorld_lerobot/agibot/RI_task_4439_458713_458713
lerobot-dataset-viz --repo-id agibot/RI_task_4439_458713_458713 \
  --root "$DS" --mode local --episode-index 0 --display-compressed-images
```

* `--display-compressed-images` avoids OOM on long episodes (it logs encoded
  frames instead of decoding every frame to raw RGB).
* On a headless box add `--save 1 --output-dir out/` to write a `.rrd`, then open
  it on a desktop with `rerun out/<name>_episode_0.rrd`.

### `head_depth` shows up black in `lerobot-dataset-viz`

That tool logs **every** camera as `rr.Image`, so depth (16-bit, values in the
bottom few % of range) renders almost black — this is a viewer limitation, not
bad data. Use the included depth-aware viewer, which logs depth cameras as
`rr.DepthImage` (rerun then colormaps + auto-ranges them):

```bash
# spawn the rerun viewer (RGB + a properly colorized depth view)
python -m agibot2lerobot.viz_depth ./AgibotWorld_lerobot/agibot/RI_task_4439_458713_458713 --episode 0

# headless: write a .rrd (use --max-frames N for a quick look)
python -m agibot2lerobot.viz_depth <dataset_dir> --episode 0 --save out/ --max-frames 200
```
(simulation datasets have no depth, so this behaves like the normal viewer there.)

---

## 5. Depth (`head_depth`) — read it correctly

`head_depth` (ImitationLearning / RichInteraction only — simulation has no depth)
is a **16-bit** `gray16be` video. In the viewer it looks almost black because
real depths occupy the bottom few % of the 0–65535 range — **this is normal, the
data is intact**. But lerobot's standard image path decodes depth to **8-bit**,
losing precision. For metric depth, read the 16-bit video directly:

```python
from agibot2lerobot.depth import read_depth_uint16, colorize_depth
depth = read_depth_uint16("AgibotWorld_lerobot/agibot/RI_task_4439_458713_458713")
print(depth.shape, depth.dtype, depth.min(), depth.max())   # (N,H,W) uint16 (mm)

# dump a viewable colormapped PNG of frame 0:
#   python -m agibot2lerobot.depth <dataset_dir> --frame 0 --out depth0.png
```

---

## Datasets / cameras

| Source | Cameras |
|--------|---------|
| ImitationLearning, RichInteraction (real) | top_head, hand_left, hand_right, **head_depth**, head_left/right/back_fisheye (7) |
| simulation | top_head, hand_left, hand_right (3, **no depth**) |

## Tested with

Python 3.12 · lerobot 0.5.2 · torch 2.11 · torchcodec 0.11 · ffmpeg 7.1 ·
rerun-sdk 0.26 (Linux). If PyPI lacks `lerobot==0.5.2`, install from source at the
matching tag and re-run `pip install -e .` here.

## License

MIT (this repo). AgiBot World data is CC BY-NC-SA 4.0 — see the dataset page.
