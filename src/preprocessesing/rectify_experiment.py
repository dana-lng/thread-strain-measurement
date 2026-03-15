from __future__ import annotations

import argparse
from pathlib import Path

from rectify_first_versuch1 import build_output_path, load_calibration, load_image, rectify_image
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / 'data' / 'raw'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Rectify all TIFF images of an experiment.')
    parser.add_argument('experiment', help='Experiment folder name, for example Versuch_10')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit for number of images.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = DEFAULT_INPUT_ROOT / args.experiment
    if not input_dir.exists():
        raise FileNotFoundError(f'Experiment not found: {input_dir}')

    calibration = load_calibration(ROOT / 'data' / 'calibration' / 'camera_calibration.json')
    files = sorted(input_dir.glob('*.tif'))
    if args.limit is not None:
        files = files[:args.limit]

    for index, input_path in enumerate(files, start=1):
        output_path = build_output_path(input_path.resolve(), None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = load_image(input_path)
        rectified = rectify_image(image, calibration)
        Image.fromarray(rectified, mode='L').save(output_path)
        if index == 1 or index % 250 == 0 or index == len(files):
            print(f'{args.experiment}: {index}/{len(files)} -> {output_path.name}')


if __name__ == '__main__':
    main()
