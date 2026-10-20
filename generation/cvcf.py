"""Step 8. Cross-View Consistency Filtering (CVCF, Sec. 3.3).

Every patch of a synthetic view is matched to its most similar patch of the source image in frozen DINOv2 features
(Eq. 1), the similarities are averaged over the visible patches only (Eq. 2), then over the N views of the source
image (Eq. 3). The synthetic views of the top 50% source images are kept, written to <synthetic>/train_cvcf.txt.

python generation/cvcf.py AG-ReID_3D-VITAL AG-ReID

<synthetic> holds the synthetic views <image>_XX with their visibility maps <image>_XX_visibility.png and the list
train_all.txt (step 7), <real> holds the source images <image>. Both are searched recursively.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Standard DINOv2 preprocessing (shortest side to 256, center crop 224). The visibility map goes through the same
# resize and crop, so that it stays aligned with the patch grid.
RESIZE, CROP, PATCH = 256, 224, 14
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def source_of(path):
    """'P0001T04041A0C0F121_03.jpg' -> 'P0001T04041A0C0F121'."""
    return Path(path).stem.rsplit("_", 1)[0]


def resize_crop(image, resample):
    w, h = image.size
    scale = RESIZE / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), resample)
    w, h = image.size
    left, top = (w - CROP) // 2, (h - CROP) // 2
    return image.crop((left, top, left + CROP, top + CROP))


def load_image(path):
    image = resize_crop(Image.open(path).convert("RGB"), Image.BICUBIC)
    return (torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1) - MEAN) / STD


def patch_visibility(path):
    """V tilde, the fraction of visible pixels of each DINOv2 patch, flattened (P,)."""
    visibility = np.asarray(resize_crop(Image.open(path).convert("L"), Image.NEAREST)) > 127
    return F.avg_pool2d(torch.from_numpy(visibility).float()[None, None], PATCH).flatten()


def view_score(source_patches, view_patches, visibility):
    """Eq. 1-2. NaN if no patch of the view is visible, the view then does not count in Eq. 3."""
    if visibility.sum() == 0:
        return float("nan")
    similarity = F.normalize(view_patches, dim=-1) @ F.normalize(source_patches, dim=-1).T
    sigma = similarity.max(dim=1).values
    return ((visibility * sigma).sum() / visibility.sum()).item()


def keep_top(scores, keep):
    """Eq. 3, then keep the `keep` fraction of source images with the highest mean score."""
    means = {s: np.nanmean(v) for s, v in scores.items() if not np.all(np.isnan(v))}
    ranked = sorted(means, key=means.get, reverse=True)
    return set(ranked[: round(keep * len(ranked))])


@torch.inference_mode()
def patch_features(model, images, device):
    """Patch tokens of a frozen DINOv2, mean of the last 4 layers, (B, P, D)."""
    out = model(pixel_values=torch.stack(images).to(device), output_hidden_states=True)
    return torch.stack(out.hidden_states[-4:]).mean(0)[:, 1:].float().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("synthetic", type=Path, help="synthetic dataset directory, with train_all.txt")
    parser.add_argument("real", type=Path, help="real dataset directory (source images)")
    parser.add_argument("--keep", type=float, default=0.5, help="fraction of source images kept")
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=256, help="images per forward pass")
    parser.add_argument("--scores", type=Path, help="optional CSV of the per-view scores")
    args = parser.parse_args()

    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    exts = (".jpg", ".jpeg", ".png")
    sources = {p.stem: p for p in args.real.rglob("*") if p.suffix.lower() in exts}
    views = defaultdict(list)
    for p in sorted(args.synthetic.rglob("*")):
        if p.suffix.lower() in exts and not p.stem.endswith("_visibility") and source_of(p) in sources:
            vis = p.with_name(f"{p.stem}_visibility.png")
            if vis.exists():
                views[source_of(p)].append((p, vis))
    print(f"{sum(map(len, views.values()))} views with a visibility map, from {len(views)} source images")
    if not views:
        raise SystemExit(f"No synthetic view with a visibility map found in {args.synthetic}")

    scores, rows = {}, []
    names = sorted(views)
    per_batch = max(1, args.batch_size // 11)  # a source image and its ~10 views
    for start in range(0, len(names), per_batch):
        chunk = names[start : start + per_batch]
        images, offsets = [], []
        for name in chunk:
            offsets.append(len(images))
            images += [load_image(sources[name])] + [load_image(p) for p, _ in views[name]]
        features = patch_features(model, images, device)
        for name, offset in zip(chunk, offsets):
            scores[name] = np.array(
                [
                    view_score(features[offset], features[offset + k], patch_visibility(vis))
                    for k, (_, vis) in enumerate(views[name], start=1)
                ]
            )
            rows += [(p.stem, s) for (p, _), s in zip(views[name], scores[name])]
        print(f"\r{start + len(chunk)}/{len(names)} source images", end="", flush=True)
    print()

    if args.scores:
        args.scores.write_text("view,score\n" + "".join(f"{v},{s:.4f}\n" for v, s in rows))

    kept = keep_top(scores, args.keep)
    scored = {p.stem for name in kept for p, _ in views[name]}
    lines = (args.synthetic / "train_all.txt").read_text().splitlines()
    filtered = [line for line in lines if Path(line).stem in scored]
    if not filtered:
        raise SystemExit("CVCF kept no view, check that train_all.txt lists the synthetic views")
    (args.synthetic / "train_cvcf.txt").write_text("".join(f"{line}\n" for line in filtered))
    print(f"Kept {len(kept)}/{len(scores)} source images, {len(filtered)}/{len(lines)} views in train_cvcf.txt")


if __name__ == "__main__":
    main()
