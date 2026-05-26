"""Real xView3 SAR vessel detector (AI2 Skylight, Apache-2.0) wrapped for septentrion.

This wraps the pretrained ``frcnn_multihead_pseudo_softer`` Faster R-CNN from the
AI2 ``sar_vessel_detect`` repo (the model that placed in the xView3 challenge). The
model was authored for torch 1.10 / torchvision 0.11; two compatibility shims let it
build and load its checkpoint cleanly under modern torch (verified: 0 missing / 0
unexpected state-dict keys, 100% recall on HIGH-confidence vessels with ~1px
localisation on a real validation scene).

External assets (fetch with ``scripts/setup_xview3.sh``):
  - AI2 source code at ``third_party/sar_vessel_detect/src`` (Apache-2.0)
  - pretrained weights at ``data/weights/xview3-combo8.pth`` (public S3)
"""

from __future__ import annotations

import configparser
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from septentrion.detect.tiling import dedup_by_proximity, window_offsets
from septentrion.schemas import Detection

if TYPE_CHECKING:
    from septentrion.io.xview3_scene import XView3Scene

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AI2_SRC = _REPO_ROOT / "third_party" / "sar_vessel_detect" / "src"
DEFAULT_CONFIG = _REPO_ROOT / "third_party" / "sar_vessel_detect" / "data" / "configs" / "final.txt"
DEFAULT_WEIGHTS = _REPO_ROOT / "data" / "weights" / "xview3-combo8.pth"

_NUM_CLASSES = 4
_NUM_CHANNELS = 3  # vh, vv, bathymetry


def install_compat_shims() -> None:
    """Make the torch-1.10-era AI2 model build under modern torchvision.

    1. Stub ``xview3.models.yolov5`` so importing the model package doesn't pull
       ultralytics (an unrelated, heavy dependency we never use).
    2. Wrap ``resnet_fpn_backbone`` to drop the removed ``pretrained=`` kwarg
       (the full checkpoint is loaded afterwards, so backbone init weights are moot).

    Idempotent.
    """
    if "xview3.models.yolov5" not in sys.modules:
        stub = types.ModuleType("xview3.models.yolov5")
        stub.create_model = lambda *a, **k: None  # ty: ignore[unresolved-attribute]
        sys.modules["xview3.models.yolov5"] = stub

    import torchvision.models.detection.backbone_utils as bu

    if getattr(bu.resnet_fpn_backbone, "_septentrion_shim", False):
        return
    original = bu.resnet_fpn_backbone

    def shimmed(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs.setdefault("weights", None)
        kwargs.setdefault("weights_backbone", None)
        try:
            return original(*args, **kwargs)
        except TypeError:
            kwargs.pop("weights_backbone", None)
            return original(*args, **kwargs)

    shimmed._septentrion_shim = True  # ty: ignore[unresolved-attribute]
    bu.resnet_fpn_backbone = shimmed


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(
    weights_path: str | Path = DEFAULT_WEIGHTS,
    *,
    ai2_src: str | Path = DEFAULT_AI2_SRC,
    config_path: str | Path = DEFAULT_CONFIG,
    image_size: int = 3072,
    device: str | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    """Build the AI2 Faster R-CNN and load its pretrained checkpoint.

    Raises ``FileNotFoundError`` if the AI2 source, config, or weights are missing
    (run ``scripts/setup_xview3.sh``).
    """
    ai2_src = Path(ai2_src)
    config_path = Path(config_path)
    weights_path = Path(weights_path)
    for path, what in [(ai2_src, "AI2 source"), (config_path, "config"), (weights_path, "weights")]:
        if not path.exists():
            raise FileNotFoundError(f"{what} not found at {path}; run scripts/setup_xview3.sh")

    if str(ai2_src) not in sys.path:
        sys.path.insert(0, str(ai2_src))
    install_compat_shims()

    from xview3.models.frcnn_multihead_pseudo_softer import FasterRCNNmps

    config = configparser.ConfigParser()
    config.read(config_path)
    dev = _resolve_device(device)
    model = FasterRCNNmps(
        num_classes=_NUM_CLASSES,
        num_channels=_NUM_CHANNELS,
        device=dev,
        config=config["training"],
        image_size=image_size,
    )
    state = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state)
    model.to(dev).eval()
    return model, dev


class Xview3Detector:
    """Pretrained xView3 SAR vessel detector over an :class:`XView3Scene`.

    Tiles the scene into overlapping windows, runs the Faster R-CNN on each, maps
    box centres to scene pixels then WGS84 lon/lat, and deduplicates detections that
    fall within ``dedup_px`` of a higher-scoring one in an overlap region.
    """

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        *,
        ai2_src: str | Path = DEFAULT_AI2_SRC,
        config_path: str | Path = DEFAULT_CONFIG,
        device: str | None = None,
        window_size: int = 3072,
        padding: int = 400,
        score_threshold: float = 0.3,
        dedup_px: float = 10.0,
    ) -> None:
        self.window_size = window_size
        self.padding = padding
        self.score_threshold = score_threshold
        self.dedup_px = dedup_px
        self.model, self.device = load_model(
            weights_path,
            ai2_src=ai2_src,
            config_path=config_path,
            image_size=window_size,
            device=device,
        )

    def _infer_window(self, stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the model on a normalised ``(3, H, W)`` window; return (centres_rc, scores).

        ``centres_rc`` is an ``(N, 2)`` array of (row, col) box centres in window pixels.
        """
        im = torch.from_numpy(stacked).to(self.device)
        with torch.no_grad():
            out = self.model([im])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        if len(boxes) == 0:
            return np.empty((0, 2)), np.empty((0,))
        centres = np.stack(
            [(boxes[:, 1] + boxes[:, 3]) / 2.0, (boxes[:, 0] + boxes[:, 2]) / 2.0], axis=1
        )
        return centres, scores

    def detect_scene(
        self,
        scene: XView3Scene,
        scene_id: str,
        *,
        row_range: tuple[int, int] | None = None,
        col_range: tuple[int, int] | None = None,
    ) -> list[Detection]:
        """Detect vessels across (a region of) a scene, returning georeferenced detections.

        ``row_range``/``col_range`` optionally restrict inference to a sub-rectangle
        (useful for a fast partial run); default covers the whole scene.
        """
        r_lo, r_hi = row_range or (0, scene.height)
        c_lo, c_hi = col_range or (0, scene.width)

        raw: list[tuple[float, float, float]] = []  # (scene_row, scene_col, score)
        for row_off in window_offsets(scene.height, self.window_size, self.padding):
            if row_off + self.window_size <= r_lo or row_off >= r_hi:
                continue
            for col_off in window_offsets(scene.width, self.window_size, self.padding):
                if col_off + self.window_size <= c_lo or col_off >= c_hi:
                    continue
                stacked = scene.read_window(row_off, col_off, self.window_size)
                centres, scores = self._infer_window(stacked)
                for (cr, cc), sc in zip(centres, scores, strict=True):
                    if sc < self.score_threshold:
                        continue
                    raw.append((row_off + float(cr), col_off + float(cc), float(sc)))

        kept = dedup_by_proximity(raw, self.dedup_px)
        detections: list[Detection] = []
        for srow, scol, score in kept:
            lon, lat = scene.pixel_to_lonlat(srow, scol)
            detections.append(
                Detection(
                    scene_id=scene_id,
                    row=round(srow),
                    col=round(scol),
                    lon=lon,
                    lat=lat,
                    detection_confidence=min(1.0, score),
                )
            )
        return detections
