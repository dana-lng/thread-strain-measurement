from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / 'data' / 'preprocessed'


def load_grayscale_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert('L'), dtype=np.uint8)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(image, pad_width=pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    filtered = np.median(windows, axis=(-2, -1))
    return filtered.astype(np.uint8)


def build_output_path(input_path: Path, output_path: Path | None, suffix: str) -> Path:
    if output_path is not None:
        return output_path
    return DEFAULT_OUTPUT_DIR / f'{input_path.stem}{suffix}.png'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Apply median smoothing to a grayscale image.')
    parser.add_argument('input_image', type=Path, help='Path to the input TIFF/PNG image.')
    parser.add_argument('--output', type=Path, default=None, help='Optional explicit output file path.')
    parser.add_argument('--kernel-size', type=int, default=3, help='Odd median filter size, default 3.')
    parser.add_argument('--suffix', default='_smoothed', help='Suffix used when no explicit output path is given.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        raise ValueError('kernel-size must be a positive odd number.')

    input_path = args.input_image.resolve()
    output_path = build_output_path(input_path, args.output, args.suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = load_grayscale_image(input_path)
    smoothed = median_filter(image, kernel_size=args.kernel_size)
    Image.fromarray(smoothed, mode='L').save(output_path)

    print(f'Input: {input_path}')
    print(f'Output: {output_path}')
    print(f'Size: {image.shape[1]}x{image.shape[0]}')
    print(f'Kernel: {args.kernel_size}x{args.kernel_size}')


if __name__ == '__main__':
    main()
