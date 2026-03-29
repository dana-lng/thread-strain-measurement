from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib.figure import Figure
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / 'data' / 'preprocessed'
OUTPUT_ROOT = ROOT / 'figures' / 'sobel'
TRACKING_MARGIN = 28
TRACKING_SCORE_RATIO = 0.7
MIN_EXPECTED_DISTANCE_RATIO = 0.08
MAX_EXPECTED_DISTANCE_RATIO = 0.22
ROI_X_START_RATIO = 0.20
ROI_X_END_RATIO = 0.80
TOP_CANDIDATES = 25


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert('L'), dtype=np.float32)


def sobel_y(image: np.ndarray) -> np.ndarray:
    # Direkte Nachbarzugriffe sind hier deutlich schneller als sliding_window_view.
    padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    top = padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:]
    bottom = padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    return top - bottom


def smooth_1d(values: np.ndarray, radius: int = 9) -> np.ndarray:
    # Glaettet 1D-Profile, damit kleine lokale Schwankungen weniger stoeren.
    kernel = np.ones(radius, dtype=np.float32) / radius
    padded = np.pad(values, (radius // 2, radius // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def local_minima(values: np.ndarray) -> np.ndarray:
    center = values[1:-1]
    left = values[:-2]
    right = values[2:]
    return np.where((center <= left) & (center < right))[0] + 1


def local_maxima(values: np.ndarray) -> np.ndarray:
    center = values[1:-1]
    left = values[:-2]
    right = values[2:]
    return np.where((center >= left) & (center > right))[0] + 1


def _score_rows(candidate_rows: np.ndarray, intensity_profile: np.ndarray, sobel_profile: np.ndarray) -> np.ndarray:
    # Fuer diesen Test entscheidet nur die Peak-Hoehe im Sobel-Profil.
    return sobel_profile[candidate_rows]


def _distance_is_plausible(markers: np.ndarray, height: int) -> bool:
    distance = int(markers[1] - markers[0])
    min_distance = int(height * MIN_EXPECTED_DISTANCE_RATIO)
    max_distance = int(height * MAX_EXPECTED_DISTANCE_RATIO)
    return min_distance <= distance <= max_distance


def _build_peak_candidates(intensity_profile: np.ndarray, sobel_profile: np.ndarray, valid_rows: np.ndarray) -> np.ndarray:
    # Kandidaten kommen nur aus lokalen Maxima des Sobel-Profils.
    candidates = local_maxima(sobel_profile)
    candidates = candidates[(candidates >= valid_rows[0]) & (candidates <= valid_rows[-1])]
    if len(candidates) == 0:
        return valid_rows
    return candidates


def _pick_marker_near_target(
    candidate_rows: np.ndarray,
    intensity_profile: np.ndarray,
    sobel_profile: np.ndarray,
    target: int,
    tracking_margin: int,
) -> tuple[int | None, float]:
    # Fuer Folgebilder bleiben wir strikt im Fenster um den letzten Marker.
    local_rows = candidate_rows[np.abs(candidate_rows - target) <= tracking_margin]
    if len(local_rows) == 0:
        lower = max(0, int(target - tracking_margin))
        upper = min(len(sobel_profile) - 1, int(target + tracking_margin))
        if upper < lower:
            return None, float('-inf')
        local_rows = np.arange(lower, upper + 1, dtype=int)

    local_scores = _score_rows(local_rows, intensity_profile, sobel_profile)
    if len(local_scores) == 0:
        return None, float('-inf')
    distance_penalty = np.abs(local_rows - target) / max(tracking_margin, 1)
    combined = local_scores - 0.35 * local_scores.max(initial=0.0) * distance_penalty
    best_idx = int(np.argmax(combined))
    return int(local_rows[best_idx]), float(local_scores[best_idx])


def _best_peak_pair(candidate_rows: np.ndarray, intensity_profile: np.ndarray, sobel_profile: np.ndarray, height: int, min_spacing: int) -> np.ndarray | None:
    if len(candidate_rows) < 2:
        return None

    scores = _score_rows(candidate_rows, intensity_profile, sobel_profile)
    order = np.argsort(scores)[::-1][: min(TOP_CANDIDATES, len(candidate_rows))]
    top_rows = candidate_rows[order]
    top_scores = scores[order]

    expected_distance = 0.5 * (height * MIN_EXPECTED_DISTANCE_RATIO + height * MAX_EXPECTED_DISTANCE_RATIO)
    best_pair: np.ndarray | None = None
    best_value = float('-inf')

    for i in range(len(top_rows)):
        for j in range(i + 1, len(top_rows)):
            pair = np.array(sorted([int(top_rows[i]), int(top_rows[j])]), dtype=int)
            distance = int(pair[1] - pair[0])
            if distance < min_spacing or not _distance_is_plausible(pair, height):
                continue

            distance_penalty = abs(distance - expected_distance) / max(expected_distance, 1.0)
            pair_value = float(top_scores[i] + top_scores[j] - 0.35 * max(top_scores[i], top_scores[j]) * distance_penalty)
            if pair_value > best_value:
                best_value = pair_value
                best_pair = pair

    return best_pair


def find_two_markers(image: np.ndarray, min_spacing: int = 80, previous_markers: np.ndarray | None = None, tracking_margin: int = TRACKING_MARGIN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    x0 = max(0, int(width * ROI_X_START_RATIO))
    x1 = min(width, int(width * ROI_X_END_RATIO))
    roi = image[:, x0:x1]

    intensity_profile = smooth_1d(roi.mean(axis=1), radius=11)
    sobel_profile = smooth_1d(np.abs(sobel_y(roi)).mean(axis=1), radius=11)

    margin = max(30, int(height * 0.05))
    valid_rows = np.arange(margin, height - margin)
    peak_candidates = _build_peak_candidates(intensity_profile, sobel_profile, valid_rows)

    if previous_markers is not None and len(previous_markers) == 2:
        tracked: list[int] = []
        for target in np.sort(previous_markers.astype(int)):
            picked, _ = _pick_marker_near_target(peak_candidates, intensity_profile, sobel_profile, int(target), tracking_margin)
            if picked is not None:
                tracked.append(picked)

        if len(tracked) == 2:
            tracked_arr = np.array(sorted(tracked), dtype=int)
            if _distance_is_plausible(tracked_arr, height):
                return tracked_arr, intensity_profile, sobel_profile

        previous_arr = np.array(sorted(previous_markers.astype(int)), dtype=int)
        previous_arr = np.clip(previous_arr, valid_rows[0], valid_rows[-1])
        return previous_arr, intensity_profile, sobel_profile

    best_pair = _best_peak_pair(peak_candidates, intensity_profile, sobel_profile, height, min_spacing)
    if best_pair is not None:
        return best_pair, intensity_profile, sobel_profile

    raise RuntimeError('Es konnten nicht zwei plausiblen Marker-Peaks gefunden werden.')


def save_sobel_diagram(
    image: np.ndarray,
    markers: np.ndarray,
    intensity_profile: np.ndarray,
    sobel_profile: np.ndarray,
    input_path: Path,
    input_root: Path = INPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    relative_parent = input_path.parent.relative_to(input_root)
    output_dir = output_root / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{input_path.stem}_sobel_diagram.png'

    fig = Figure(figsize=(16, 9), dpi=150)
    axes = fig.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1.35]})
    axes[0].imshow(image, cmap='gray', aspect='auto')
    axes[0].set_title('Bild')
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


def process_image(path: Path, previous_markers: np.ndarray | None = None) -> tuple[Path, np.ndarray]:
    image = load_image(path)
    markers, intensity_profile, sobel_profile = find_two_markers(image, previous_markers=previous_markers)
    output_path = save_sobel_diagram(image, markers, intensity_profile, sobel_profile, path)
    print(f'{path.name}: Marker bei y={markers[0]} und y={markers[1]}')
    print(f'Diagramm: {output_path}')
    return output_path, markers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Wendet Sobel auf Bilder an und markiert zwei dunkle Markierungen.')
    parser.add_argument('--input', type=Path, default=INPUT_ROOT, help='Einzelbild oder Ordner unter data/preprocessed')
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
