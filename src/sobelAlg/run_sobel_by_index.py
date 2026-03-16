from __future__ import annotations

import argparse
from pathlib import Path

from detect_markers import INPUT_ROOT, process_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Wendet den Sobel-Markierungsalgorithmus auf das n-te Bild eines Versuchs an.',
    )
    parser.add_argument('experiment', help='Versuchsordner, zum Beispiel Versuch_1 oder Versuch_10')
    parser.add_argument('image_index', type=int, help='1-basierter Index des Bildes im Versuch')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image_index < 1:
        raise ValueError('image_index muss >= 1 sein.')

    experiment_dir = INPUT_ROOT / args.experiment
    if not experiment_dir.exists():
        raise FileNotFoundError(f'Versuch nicht gefunden: {experiment_dir}')

    files = sorted(experiment_dir.glob('*.png'))
    if not files:
        raise FileNotFoundError(f'Keine PNG-Bilder gefunden in: {experiment_dir}')

    if args.image_index > len(files):
        raise IndexError(
            f'image_index={args.image_index} ist zu gross. ' 
            f'Verfuegbar sind {len(files)} Bilder in {args.experiment}.'
        )

    input_path = files[args.image_index - 1]
    print(f'Versuch: {args.experiment}')
    print(f'Bild #{args.image_index}: {input_path.name}')
    process_image(input_path)


if __name__ == '__main__':
    main()
