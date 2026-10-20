---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: image-feature-extraction
tags:
  - person-re-identification
  - aerial-ground
  - vision-transformer
  - neurips2026
---

# 3D-VITAL

Checkpoints of **3D-VITAL: Visibility-aware Identity Training from Artificial Liftings of 3D Reconstruction model for
Cross-View Object Re-Identification** (NeurIPS 2026): [arXiv:2610.16692](https://arxiv.org/abs/2610.16692),
[poster](https://neurips.cc/virtual/2026/loc/paris/poster/150244).
Code, training and evaluation are at <https://github.com/TurtleSmoke/3D-VITAL>.

## Checkpoints

Each checkpoint is trained on one camera type of AG-ReID, given by its name.

| Checkpoint          | Query → Gallery | mAP   | Rank-1 |
|---------------------|-----------------|-------|--------|
| `agreid_aerial.pth` | A → G           | 52.09 | 66.20  |
|                     | G → A           | 53.30 | 62.68  |
| `agreid_ground.pth` | A → G           | 55.99 | 67.13  |
|                     | G → A           | 57.93 | 68.61  |

A = aerial, G = ground. The checkpoints are stored in fp16, so they can differ slightly from the paper.

## Usage

```bash
git clone https://github.com/TurtleSmoke/3D-VITAL && cd 3D-VITAL && uv sync && source .venv/bin/activate
hf download TurtleSmoke/3D-VITAL --include "*.pth" --local-dir checkpoints
bash scripts/eval.sh agreid_aerial $DATA checkpoints/agreid_aerial.pth
```

## License

The weights are released for non-commercial research under CC BY-NC 4.0. They were trained on
[AG-ReID](https://github.com/huynguyen792/AG-ReID) and on synthetic views generated with
[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) and [StyleID](https://github.com/jiwoogit/StyleID), whose
respective terms also apply. The code is under the CeCILL-B license.

## Citation

```bibtex
@inproceedings{grolleau2026vital,
    title = {3D-VITAL: Visibility-aware Identity Training from Artificial Liftings of 3D Reconstruction model for Cross-View Object Re-Identification},
    author = {Grolleau, William and Sabourin, Astrid and Lapouge, Guillaume and Achard, Catherine},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2026}
}
```
