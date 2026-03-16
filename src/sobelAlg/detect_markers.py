from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / 'data' / 'preprocessed'
OUTPUT_ROOT = ROOT / 'figures' / 'sobel'
TRACKING_MARGIN = 40


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert('L'), dtype=np.float32)


def sobel_y(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    kernel = np.array(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        dtype=np.float32,
    )
    return np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))


def smooth_1d(values: np.ndarray, radius: int = 9) -> np.ndarray:
    kernel = np.ones(radius, dtype=np.float32) / radius
    padded = np.pad(values, (radius // 2, radius // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def local_minima(values: np.ndarray) -> np.ndarray:
    center = values[1:-1]
    left = values[:-2]
    right = values[2:]
    return np.where((center <= left) & (center < right))[0] + 1


def _score_rows(
    candidate_rows: np.ndarray,
    intensity_profile: np.ndarray,
    sobel_profile: np.ndarray,
) -> np.ndarray:
    darkness = 255.0 - intensity_profile[candidate_rows]
    edge_strength = sobel_profile[candidate_rows]
    return darkness + 0.5 * edge_strength


def _pick_marker_near_target(
    candidate_rows: np.ndarray,
    intensity_profile: np.ndarray,
    sobel_profile: np.ndarray,
    target: int,
    tracking_margin: int,
) -> int | None:
    local_rows = candidate_rows[np.abs(candidate_rows - target) <= tracking_margin]
    if len(local_rows) == 0:
        return None

    local_scores = _score_rows(local_rows, intensity_profile, sobel_profile)
    distance_penalty = np.abs(local_rows - target) / max(tracking_margin, 1)
    combined = local_scores - 0.35 * local_scores.max(initial=0.0) * distance_penalty
    return int(local_rows[np.argmax(combined)])


def find_two_markers(
    image: np.ndarray,
    min_spacing: int = 80,
    previous_markers: np.ndarray | None = None,
    tracking_margin: int = TRACKING_MARGIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    x0 = max(0, int(width * 0.2))
    x1 = min(width, int(width * 0.8))
    roi = image[:, x0:x1]

    intensity_profile = smooth_1d(roi.mean(axis=1), radius=11)
    sobel_profile = smooth_1d(np.abs(sobel_y(roi)).mean(axis=1), radius=11)

    margin = max(30, int(height * 0.05))
    valid_rows = np.arange(margin, height - margin)
    minima = local_minima(intensity_profile)
    minima = minima[(minima >= margin) & (minima < height - margin)]
    if len(minima) == 0:
        minima = valid_rows

    if previous_markers is not None and len(previous_markers) == 2:
        tracked: list[int] = []
        for target in np.sort(previous_markers.astype(int)):
            picked = _pick_marker_near_target(minima, intensity_profile, sobel_profile, int(target), tracking_margin)
            if picked is None:
                picked = _pick_marker_near_target(valid_rows, intensity_profile, sobel_profile, int(target), tracking_margin)
            if picked is not None:
                tracked.append(picked)

        if len(tracked) == 2 and abs(tracked[1] - tracked[0]) >= min_spacing // 2:
            return np.array(sorted(tracked), dtype=int), intensity_profile, sobel_profile

    score = _score_rows(minima, intensity_profile, sobel_profile)
    order = minima[np.argsort(score)[::-1]]

    selected: list[int] = []
    for row in order:
        if all(abs(int(row) - existing) >= min_spacing for existing in selected):
            selected.append(int(row))
        if len(selected) == 2:
            break

    if len(selected) < 2:
        fallback_scores = _score_rows(valid_rows, intensity_profile, sobel_profile)
        fallback = valid_rows[np.argsort(fallback_scores)[::-1]]
        for row in fallback:
            if all(abs(int(row) - existing) >= min_spacing for existing in selected):
                selected.append(int(row))
            if len(selected) == 2:
                break

    if len(selected) < 2:
        raise RuntimeError('Es konnten nicht zwei Markierungen gefunden werden.')

    return np.array(sorted(selected), dtype=int), intensity_profile, sobel_profile


def save_sobel_diagram(
    image: np.ndarray,
    markers: np.ndarray,
    intensity_profile: np.ndarray,
    sobel_profile: np.ndarray,
    input_path: Path,
) -> Path:
    relative_parent = input_path.parent.relative_to(INPUT_ROOT)
    output_dir = OUTPUT_ROOT / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{input_path.stem}_sobel_diagram.png'

    fig = Figure(figsize=(16, 9), dpi=150)
    axes = fig.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1.35]})

    axes[0].imshow(image, cmap='gray', aspect='auto')
    axes[0].set_title('Preprocessed Bild')
    axes[0].set_xlabel('X [px]')
    axes[0].set_ylabel('Y [px]')

    axes[1].imshow(image, cmap='gray', aspect='auto')
    for marker in markers:
        axes[1].axhline(marker, color='red', linewidth=2)
    axes[1].set_title('Bild mit Markierungen')
    axes[1].set_xlabel('X [px]')
    axes[1].set_ylabel('Y [px]')

    y = np.arange(image.shape[0])
    axes[2].plot(intensity_profile, y, label='Intensitaetsprofil', color='black')
    axes[2].plot(sobel_profile, y, label='|Sobel Y| Profil', color='royalblue')
    for index, marker in enumerate(markers, start=1):
        axes[2].axhline(marker, color='red', linestyle='--', linewidth=1.5, label=f'Markierung {index}')
    axes[2].invert_yaxis()
    axes[2].set_title(input_path.stem)
    axes[2].set_xlabel('Profilwert')
    axes[2].set_ylabel('Y [px]')

    handles, labels = axes[2].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    axes[2].legend(unique.values(), unique.keys(), loc='best')

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    return output_path


def plot_detection(
    image: np.ndarray,
    markers: np.ndarray,
    intensity_profile: np.ndarray,
    sobel_profile: np.ndarray,
    input_path: Path,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1, 1.35]})

    axes[0].imshow(image, cmap='gray', aspect='auto')
    axes[0].set_title('Glaettetes Bild')
    axes[0].set_xlabel('X [px]')
    axes[0].set_ylabel('Y [px]')

    axes[1].imshow(image, cmap='gray', aspect='auto')
    for marker in markers:
        axes[1].axhline(marker, color='red', linewidth=2)
    axes[1].set_title('Bild mit Markierungen')
    axes[1].set_xlabel('X [px]')
    axes[1].set_ylabel('Y [px]')

    y = np.arange(image.shape[0])
    axes[2].plot(intensity_profile, y, label='Intensitaetsprofil', color='black')
    axes[2].plot(sobel_profile, y, label='|Sobel Y| Profil', color='royalblue')
    for index, marker in enumerate(markers, start=1):
        axes[2].axhline(marker, color='red', linestyle='--', linewidth=1.5, label=f'Markierung {index}')
    axes[2].invert_yaxis()
    axes[2].set_title(input_path.stem)
    axes[2].set_xlabel('Profilwert')
    axes[2].set_ylabel('Y [px]')

    handles, labels = axes[2].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    axes[2].legend(unique.values(), unique.keys(), loc='best')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_image(path: Path, previous_markers: np.ndarray | None = None) -> tuple[Path, np.ndarray]:
    image = load_image(path)
    markers, intensity_profile, sobel_profile = find_two_markers(image, previous_markers=previous_markers)

    output_path = save_sobel_diagram(image, markers, intensity_profile, sobel_profile, path)
    print(f'{path.name}: Marker bei y={markers[0]} und y={markers[1]}')
    print(f'Diagramm: {output_path}')
    return output_path, markers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Wendet Sobel auf geglaettete Bilder an und markiert zwei dunkle Markierungen.',
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=INPUT_ROOT,
        help='Einzelbild oder Ordner unter data/preprocessed',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()

    if input_path.is_file():
        process_image(input_path)
        return

    files = sorted(input_path.rglob('*.png'))
    if not files:
        raise FileNotFoundError(f'Keine PNG-Dateien gefunden unter: {input_path}')

    previous_markers: np.ndarray | None = None
    for path in files:
        _, previous_markers = process_image(path, previous_markers=previous_markers)


if __name__ == '__main__':
    main()
