from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from PIL import Image

from rectify_first_versuch1 import build_output_path, load_calibration, load_image, preprocess_image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / 'data' / 'raw'
CALIBRATION_PATH = ROOT / 'data' / 'calibration' / 'camera_calibration.json'


ProgressCallback = Callable[[int, int, Path], None]
ShouldCancelCallback = Callable[[], bool]


class ProcessingCancelled(Exception):
    pass


def process_experiment(
    experiment: str,
    limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
    should_cancel: ShouldCancelCallback | None = None,
) -> int:
    input_dir = DEFAULT_INPUT_ROOT / experiment
    if not input_dir.exists():
        raise FileNotFoundError(f'Experiment not found: {input_dir}')

    calibration = load_calibration(CALIBRATION_PATH)
    files = sorted(input_dir.glob('*.tif'))
    if not files:
        raise FileNotFoundError(f'No TIFF images found in: {input_dir}')

    if limit is not None:
        files = files[:limit]

    total = len(files)
    for index, input_path in enumerate(files, start=1):
        if should_cancel is not None and should_cancel():
            raise ProcessingCancelled(f'Preprocessing von {experiment} wurde abgebrochen.')

        output_path = build_output_path(input_path.resolve(), None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = load_image(input_path)
        processed = preprocess_image(image, calibration)
        Image.fromarray(processed, mode='L').save(output_path)

        if progress_callback is not None:
            progress_callback(index, total, output_path)
        elif index == 1 or index % 250 == 0 or index == total:
            print(f'{experiment}: {index}/{total} -> {output_path.name}')

    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocess all TIFF images of an experiment.')
    parser.add_argument('experiment', help='Experiment folder name, for example Versuch_10')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit for number of images.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_experiment(args.experiment, limit=args.limit)


if __name__ == '__main__':
    main()
