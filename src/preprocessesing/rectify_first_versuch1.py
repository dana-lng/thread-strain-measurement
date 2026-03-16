from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_FILE = ROOT / 'data' / 'calibration' / 'camera_calibration.json'
DEFAULT_OUTPUT_ROOT = ROOT / 'data' / 'preprocessed'


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert('L'), dtype=np.uint8)


def load_calibration(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(image, pad_width=pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    filtered = np.median(windows, axis=(-2, -1))
    return filtered.astype(np.uint8)


def rectify_image(image: np.ndarray, calibration: dict) -> np.ndarray:
    height, width = image.shape
    calib_height = float(calibration['image_height'])
    shear = calibration['shear_model']
    slope_a = float(shear['slope_a'])
    slope_b = float(shear['slope_b'])
    u_center = float(calibration.get('u_center', width / 2.0))

    dst_x, dst_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    y_scale = height / calib_height if calib_height > 0 else 1.0
    dst_y_cal = dst_y / max(y_scale, 1e-6)
    slope_cal = slope_a * dst_y_cal + slope_b
    slope_current = slope_cal * y_scale
    src_y = dst_y + slope_current * (dst_x - u_center)

    y0 = np.floor(src_y).astype(np.int32)
    y1 = y0 + 1
    x = dst_x.astype(np.int32)

    valid = (y0 >= 0) & (y1 < height)
    y0_clipped = np.clip(y0, 0, height - 1)
    y1_clipped = np.clip(y1, 0, height - 1)

    top = image[y0_clipped, x].astype(np.float32)
    bottom = image[y1_clipped, x].astype(np.float32)
    weight = src_y - y0

    remapped = top * (1.0 - weight) + bottom * weight
    remapped[~valid] = 128.0
    return np.clip(remapped, 0, 255).astype(np.uint8)


def preprocess_image(image: np.ndarray, calibration: dict, kernel_size: int = 3) -> np.ndarray:
    rectified = rectify_image(image, calibration)
    return median_filter(rectified, kernel_size=kernel_size)


def build_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path

    parent_name = input_path.parent.name
    target_dir = DEFAULT_OUTPUT_ROOT / parent_name
    return target_dir / f'{input_path.stem}_rectified.png'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Apply calibrated rectification and smoothing to an input image.')
    parser.add_argument('input_image', type=Path, help='Path to the input TIFF/PNG image.')
    parser.add_argument('--output', type=Path, default=None, help='Optional explicit output file path.')
    parser.add_argument('--calibration', type=Path, default=CALIBRATION_FILE, help='Calibration JSON path.')
    parser.add_argument('--kernel-size', type=int, default=3, help='Odd median filter size, default 3.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        raise ValueError('kernel-size must be a positive odd number.')

    input_path = args.input_image.resolve()
    output_path = build_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    calibration = load_calibration(args.calibration.resolve())
    processed = preprocess_image(image, calibration, kernel_size=args.kernel_size)
    Image.fromarray(processed, mode='L').save(output_path)

    print(f'Input: {input_path}')
    print(f'Output: {output_path}')
    print(f'Size: {image.shape[1]}x{image.shape[0]}')
    print(f'Kernel: {args.kernel_size}x{args.kernel_size}')


if __name__ == '__main__':
    main()
