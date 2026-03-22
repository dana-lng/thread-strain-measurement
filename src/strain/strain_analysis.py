from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = ROOT / 'data' / 'calibration' / 'camera_calibration.json'
RESULTS_ROOT = ROOT / 'results' / 'strain'
REST_DISTANCE_MM = 20.0


@dataclass(frozen=True)
class StrainResult:
    y_top_px: float
    y_bot_px: float
    y_center_px: float
    distance_px: float
    mm_per_px: float
    distance_mm: float
    strain_percent: float


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _scaled_v_position(v_px: float, image_height: int, calib: dict[str, Any]) -> float:
    # Das Polynom wurde auf der Kalibrierbildhoehe bestimmt und wird darauf umgerechnet.
    calib_height = float(calib.get('image_height', image_height))
    if image_height <= 0:
        raise ValueError('Die Bildhoehe muss positiv sein.')
    return float(v_px) * calib_height / float(image_height)


def mm_per_px(v_px: float, image_height: int, calib: dict[str, Any]) -> float:
    coeffs = np.asarray(calib['scale_v_model']['coeffs'], dtype=np.float64)
    v_model = _scaled_v_position(v_px, image_height, calib)
    return float(np.polyval(coeffs, v_model))


def pixel_to_mm(u_px: float, v_px: float, homography: list[list[float]]) -> tuple[float, float]:
    h = np.asarray(homography, dtype=np.float64)
    px = np.array([u_px, v_px, 1.0], dtype=np.float64)
    world = h @ px
    world /= world[2]
    return float(world[0]), float(world[1])


def mm_to_pixel(x_mm: float, y_mm: float, homography: list[list[float]]) -> tuple[float, float]:
    h = np.asarray(homography, dtype=np.float64)
    inv_h = np.linalg.inv(h)
    world = np.array([x_mm, y_mm, 1.0], dtype=np.float64)
    px = inv_h @ world
    px /= px[2]
    return float(px[0]), float(px[1])


def compute_strain(markers: np.ndarray, image_height: int, calib: dict[str, Any], rest_distance_mm: float = REST_DISTANCE_MM) -> StrainResult:
    ordered = np.sort(np.asarray(markers, dtype=np.float64))
    y_top_px = float(ordered[0])
    y_bot_px = float(ordered[1])
    y_center_px = 0.5 * (y_top_px + y_bot_px)
    distance_px = abs(y_bot_px - y_top_px)
    scale = mm_per_px(y_center_px, image_height, calib)
    distance_mm = distance_px * scale
    strain_percent = (distance_mm - rest_distance_mm) / rest_distance_mm * 100.0
    return StrainResult(
        y_top_px=y_top_px,
        y_bot_px=y_bot_px,
        y_center_px=y_center_px,
        distance_px=distance_px,
        mm_per_px=scale,
        distance_mm=distance_mm,
        strain_percent=strain_percent,
    )


def ensure_results_dir(experiment: str) -> Path:
    target_dir = RESULTS_ROOT / experiment
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def open_strain_csv(experiment: str):
    target_dir = ensure_results_dir(experiment)
    output_path = target_dir / 'strain_results.csv'
    handle = output_path.open('w', encoding='utf-8', newline='')
    writer = csv.writer(handle)
    writer.writerow([
        'frame_index',
        'filename',
        'y_top_px',
        'y_bot_px',
        'y_center_px',
        'distance_px',
        'mm_per_px',
        'distance_mm',
        'strain_percent',
    ])
    return handle, writer, output_path
