from pathlib import Path

from PIL import Image
from torch.utils.data.dataset import Dataset


def load_image(path):
    """Load an image as RGB. RGBA images (segmented MOO renders) are composited on black."""
    with Image.open(path) as img:
        img = img.copy()
    if img.mode == "RGBA":
        black = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(black, img)
    return img.convert("RGB")


def load_visibility(path, domain, size):
    """Load the visibility map <view>_visibility.png of a synthetic view. Real images are entirely visible."""
    if domain == 0:
        return Image.new("L", size, 255)
    path = Path(path)
    with Image.open(path.with_name(f"{path.stem}_visibility.png")) as vis:
        vis = vis.convert("L")
    if vis.size != size:
        raise ValueError(f"The visibility map of {path} does not match its size")
    return vis


class ImageDataset(Dataset):
    """data is a list of (path, domain, pid, camid), with domain = 0 for real images and 1 for synthetic views."""

    def __init__(self, data, transform, name, visibility=False):
        self.data = data
        self.transform = transform
        self.name = name
        self.visibility = visibility  # training set, the transform also takes and returns the visibility map
        self.pids = [d[2] for d in data]
        self.num_classes = len(set(self.pids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, domain, pid, camid = self.data[idx]
        item = {
            "x": load_image(path),
            "pid": pid,
            "camid": camid,
            "domain_label": domain,
            "path": str(path),
            "filename": Path(path).stem,
        }
        if self.visibility:
            item["x"], item["vis"] = self.transform(item["x"], load_visibility(path, domain, item["x"].size))
        else:
            item["x"] = self.transform(item["x"])
        return item
