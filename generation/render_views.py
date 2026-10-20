"""Step 4. Render geometrically-consistent novel views of each calibrated mesh and their visibility maps (Sec. 3.2).

Meshes whose calibrated IoU is below tau_iou = 0.7 are discarded. Each remaining mesh is rendered N = 10 times
from its calibrated pose shifted by
    - an elevation shift linearly spaced in [0, 30] deg toward the other viewpoint-domain
      (ground source -> camera moves up, aerial source -> camera moves down), view _00 is the source elevation,
    - an azimuth shift sampled uniformly in [-60, 60] deg.
Renders are foreground-only RGBA PNGs saved as <output>/<relative mesh path>_XX.png

The first triangle hit by each pixel ray of the source camera is visible, the others are hallucinated. This visibility
mesh is rendered with the camera of each view as <output>/<relative mesh path>_XX_visibility.png (255 = visible).

python generation/render_views.py --mesh-root meshes/AG-ReID --output AG-ReID_renders meshes/AG-ReID/**/*.glb
"""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import binary_erosion  # noqa: E402

from calibrate_pose import get_camera_pose  # noqa: E402

IOU_THRESHOLD = 0.7
NUM_VIEWS = 10
MAX_ELEVATION_SHIFT = 30
MAX_AZIMUTH_SHIFT = 60
RESOLUTION = 1024


def is_aerial(path):
    """Viewpoint-domain of the source image. AG-ReID / AG-ReID.v2 camera C0 is the UAV, MOO top views are frames >= 64."""
    name = Path(path).name
    match = re.search(r"cow\d+_(\d+)", name)
    if match:
        return int(match.group(1)) >= 64
    return "C0" in name


def add_lights(scene, cam_pose):
    key = np.copy(cam_pose)
    key[:3, 3] += [0.5, 0.5, 0.0]
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0), pose=key)
    fill = np.copy(cam_pose)
    fill[:3, 3] += [-0.3, 0.0, -0.3]
    scene.add(pyrender.DirectionalLight(color=[0.8, 0.8, 0.9], intensity=1.5), pose=fill)
    rim = np.copy(cam_pose)
    rim[:3, 3] += [0.0, -0.5, -0.3]
    scene.add(
        pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=1.0, innerConeAngle=np.pi / 4, outerConeAngle=np.pi / 2),
        pose=rim,
    )


def source_visible_faces(mesh, cam_pose, fov, ar):
    """Triangles hit first by the pixel rays of the camera."""
    tan = np.tan(np.deg2rad(fov) / 2)
    ndc = (np.arange(RESOLUTION) + 0.5) / RESOLUTION * 2 - 1
    x, y = np.meshgrid(ndc * tan * ar, -ndc * tan)  # image rows go down, camera y goes up
    directions = np.stack([x.ravel(), y.ravel(), -np.ones(x.size)], axis=1) @ cam_pose[:3, :3].T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    hit = mesh.ray.intersects_first(np.tile(cam_pose[:3, 3], (len(directions), 1)), directions)
    return np.unique(hit[hit >= 0])


def visibility_mesh(mesh, visible_faces):
    """White visible triangles, black hallucinated ones."""
    colors = np.tile(np.array([0, 0, 0, 255], dtype=np.uint8), (len(mesh.faces), 1))
    colors[visible_faces, :3] = 255
    return trimesh.Trimesh(mesh.vertices, mesh.faces, face_colors=colors, process=False)


def render(mesh, vis_mesh, yaw, pitch, roll, radius, fov, ar, renderer):
    cam_pose = get_camera_pose(yaw, pitch, roll, radius)
    scene = pyrender.Scene(bg_color=[0, 0, 255, 0])
    add_lights(scene, cam_pose)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    scene.add_node(
        pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=ar), matrix=cam_pose)
    )
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

    # Same camera, flat shading, the threshold only resolves the anti-aliased edges
    vis_scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    vis_scene.add(pyrender.Mesh.from_trimesh(vis_mesh, smooth=False))
    vis_scene.add_node(
        pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=ar), matrix=cam_pose)
    )
    visibility, _ = renderer.render(vis_scene, flags=pyrender.RenderFlags.FLAT)

    # Remove the anti-aliased border, then crop to the object
    mask = binary_erosion(depth > 0, structure=np.ones((3, 3)))
    color = np.array(color)
    color[~mask] = 0
    color[:, :, 3] = mask * 255
    visibility = ((visibility[:, :, 0] > 127) & mask).astype(np.uint8) * 255
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    crop = np.s_[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return color[crop], visibility[crop]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("meshes", nargs="+", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    renderer = pyrender.OffscreenRenderer(viewport_width=RESOLUTION, viewport_height=RESOLUTION)

    for mesh_file in args.meshes:
        csv_path = mesh_file.with_suffix(".csv")
        if not csv_path.exists():
            print(f"No calibration for {mesh_file}, skipped")
            continue
        cam = pd.read_csv(csv_path).iloc[0]
        if cam["IOU"] < IOU_THRESHOLD:
            continue

        yaws = cam["Yaw"] + rng.uniform(-MAX_AZIMUTH_SHIFT, MAX_AZIMUTH_SHIFT, NUM_VIEWS)
        sign = -1 if is_aerial(mesh_file) else 1
        pitches = cam["Pitch"] + sign * np.linspace(0, MAX_ELEVATION_SHIFT, NUM_VIEWS)

        out_stem = args.output / mesh_file.relative_to(args.mesh_root).with_suffix("")
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.load(mesh_file, force="mesh")
        source_pose = get_camera_pose(cam["Yaw"], cam["Pitch"], cam["Roll"], cam["Radius"])
        vis_mesh = visibility_mesh(mesh, source_visible_faces(mesh, source_pose, cam["FOV"], cam["Aspect Ratio"]))
        for view, (yaw, pitch) in enumerate(zip(yaws, pitches)):
            out = Path(f"{out_stem}_{view:02d}.png")
            vis_out = Path(f"{out_stem}_{view:02d}_visibility.png")
            if out.exists() and vis_out.exists() and not args.force:
                continue
            image = render(
                mesh, vis_mesh, yaw, pitch, cam["Roll"], cam["Radius"], cam["FOV"], cam["Aspect Ratio"], renderer
            )
            if image is None:
                print(f"Empty render for {out}, skipped")
                continue
            Image.fromarray(image[0]).save(out)
            Image.fromarray(image[1]).save(vis_out)


if __name__ == "__main__":
    main()
