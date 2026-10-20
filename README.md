# 3D-VITAL

[![arXiv](https://img.shields.io/badge/arXiv-2610.16692-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.16692)
[![Poster](https://img.shields.io/badge/NeurIPS%202026-Poster-4b44ce)](https://neurips.cc/virtual/2026/loc/paris/poster/150244)
[![Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/TurtleSmoke/3D-VITAL)
[![Lint](https://github.com/TurtleSmoke/3D-VITAL/actions/workflows/lint.yml/badge.svg)](https://github.com/TurtleSmoke/3D-VITAL/actions/workflows/lint.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: CeCILL-B](https://img.shields.io/badge/License-CeCILL--B-green)](LICENSE)

Official code of **3D-VITAL: Visibility-aware Identity Training from Artificial Liftings of 3D Reconstruction model for
Cross-View Object Re-Identification** (NeurIPS 2026).

3D-VITAL is built upon [3D-LENS](https://github.com/TurtleSmoke/3D-LENS) and trains an aerial-ground re-identification
model from a single viewpoint (aerial *or* ground images only). The 3D mesh lifted from each training image also gives
the visibility map of each synthetic view, which separates the pixels seen in the source image from the hallucinated
ones. It is used to discard the broken meshes (Cross-View Consistency Filtering) and to steer the attention of the
network toward the visible regions (Visibility-Aware Attention Supervision).

```
generation/   synthetic view generation (segmentation, 3D lifting, calibration, rendering, background, StyleID, filtering)
reid/         training and evaluation
scripts/      utility scripts
```

## Installation

```bash
uv sync                          # training and evaluation (Python >= 3.10, torch 2.4), or: pip install -e .
uv sync --group generation       # + synthetic view generation (YOLO, pyrender, LaMa, DINOv2)
make -C reid                     # optional: Cython ranking, faster evaluation
uv run pre-commit install        # development: ruff check + format on every commit (same as the CI)
```

Commands below assume the environment is active (`source .venv/bin/activate`).

We use the pretrained weights on ImageNet-21K for the ViT backbone
([jx_vit_base_p16_224-80ecf9dd.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)),
which is expected in `checkpoints/` (or set `PRETRAINED=<path>`).

## Data

Each real dataset is paired with its synthetic counterpart, generated with [generation/](generation/README.md):

```
$DATA/
├── AG-ReID/                 # bounding_box_train/, query_all_c{0,3}/, bounding_box_test_all_c{0,3}/, train_all.txt
├── AG-ReID_3D-VITAL/        # bounding_box_train/<image>_XX.jpg, train_all.txt, train_cvcf.txt
├── AG-ReID.v2/              # AG-ReID.v2/{train_all,query,gallery}/, exp*.txt, train_all.txt
├── AG-ReID.v2_3D-VITAL/     # AG-ReID.v2/train_all/<id>/<image>_XX.jpg, train_all.txt, train_cvcf.txt
├── MOO/                     # output_screenshots_png/, output_screenshots_split/, train_all.txt
└── MOO_3D-VITAL/            # output_screenshots_png/<cow>/<image>_XX.png, train_all.txt, train_cvcf.txt
```

`train_all.txt` lists the training images of a directory (`bash generation/build_lists.sh <dataset> <dir>`), and
`train_cvcf.txt` the synthetic views kept by CVCF (`python generation/cvcf.py <synthetic dir> <real dir>`).
Synthetic views are named after their source image, with their visibility map next to them
(`<image>_XX_visibility.png`).

## Training

One configuration per single-view setting, in `reid/configs/train/`:

| Config                                             | Dataset    | Training images   | Evaluation    |
|----------------------------------------------------|------------|-------------------|---------------|
| `agreid_aerial` / `agreid_ground`                  | AG-ReID    | aerial / ground   | A→G, G→A      |
| `agreid_v2_aerial_cctv` / `agreid_v2_cctv`         | AG-ReID.v2 | aerial / CCTV     | A→C, C→A      |
| `agreid_v2_aerial_wearable` / `agreid_v2_wearable` | AG-ReID.v2 | aerial / wearable | A→W, W→A      |
| `moo_top` / `moo_side`                             | MOO        | top / side        | T→S, S→T, T+S |

```bash
bash scripts/train.sh agreid_aerial $DATA
```

Results are reported at the last epoch (`runs/ckpts/<config>/model_last.pth`).

## Evaluation

Evaluation only needs the real dataset. The AG-ReID checkpoints of the paper are on
[Hugging Face](https://huggingface.co/TurtleSmoke/3D-VITAL), with their results in the model card:

```bash
hf download TurtleSmoke/3D-VITAL --include "*.pth" --local-dir checkpoints
bash scripts/eval.sh agreid_aerial $DATA checkpoints/agreid_aerial.pth
```

They are stored in fp16 (inference parameters only) to halve their size, so evaluating them can differ slightly from
the paper.

## Citation

```bibtex
@inproceedings{grolleau2026vital,
    title = {3D-VITAL: Visibility-aware Identity Training from Artificial Liftings of 3D Reconstruction model for Cross-View Object Re-Identification},
    author = {Grolleau, William and Sabourin, Astrid and Lapouge, Guillaume and Achard, Catherine},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2026}
}
```

## Acknowledgements

The backbone comes from [TransReID](https://github.com/damo-cv/TransReID) and the losses, optimizer and scheduler
from [fast-reid](https://github.com/JDAI-CV/fast-reid). Synthetic views are generated with
[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2), [StyleID](https://github.com/jiwoogit/StyleID),
[LaMa](https://github.com/advimman/lama) and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), and
filtered with [DINOv2](https://github.com/facebookresearch/dinov2).

Special thanks to the authors of [fast-reid](https://github.com/JDAI-CV/fast-reid), whose open-source code was largely
used and adapted for this project.

## License

Copyright (c) 2026 Commissariat à l'énergie atomique et aux énergies alternatives (CEA).
This code is released under the [CeCILL-B](LICENSE) license. Files adapted from TransReID (MIT), fast-reid
(Apache-2.0) and StyleID (MIT) keep their original license.
