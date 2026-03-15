"""
Bildkorrektur (Rektifizierung) für schräge Kamera – Versuchsreihe 2026-02-26.

Die Kamera hat einen seitlichen Versatz von 500mm bei einer Normalen von 330mm
zur Fadenschar (Winkel ≈56,6°). Dadurch erscheinen real-horizontale Markierungslinien
in den Bildern geneigt (positive Steigung oben, negative Steigung unten im Bild).
Die Steigung ist Y-positions-abhängig und wird durch ein lineares Modell beschrieben:

    slope(v) = slope_a * v + slope_b

Korrektur per Y-Scherung (warpAffine):
    src_y = dst_y + slope(v_center) * (dst_x - u_center)

Workflow:
    1. Kalibrierung:  1_calibrate_camera.py → camera_calibration.json
    2. Anwendung:     rectify_image(img, params) oder load_and_crop_calibrated()

Funktionen:
    RectificationParams       : Dataclass mit Kalibrierungsparametern
    save/load_rectification_params : JSON-I/O
    rectify_image             : Wendet Y-Scherung auf ein Bild an
    measure_line_slope        : Misst Steigung einer Linie per gewichtetem Schwerpunkt
    detect_colored_lines      : Erkennt farbige Linien per HSV-Filter und misst Steigungen
    fit_slope_model           : Linearer Fit slope(v) = slope_a * v + slope_b
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Kalibrierungsparameter
# ---------------------------------------------------------------------------

@dataclass
class RectificationParams:
    """
    Kalibrierungsparameter für die Y-Scherungskorrektur.

    Attributes
    ----------
    slope_a : float
        Änderung der Scherungssteigung pro Pixel in Y-Richtung (lineare Abhängigkeit).
        Einheit: [1/px] (dimensionslos, da slope selbst dimensionslos ist)
    slope_b : float
        Scherungssteigung bei v=0 (oberer Bildrand).
        Positiv = obere Linien neigen nach rechts-unten.
    u_center : float
        X-Mitte des Bildes (Scherungszentrum). Pixel, die auf dieser Spalte liegen,
        werden nicht verschoben. Typischerweise = Bildbreite / 2 = 64.
    image_width : int
        Erwartete Bildbreite in Pixeln (zur Validierung).
    image_height : int
        Erwartete Bildhöhe in Pixeln (zur Validierung).
    """
    slope_a: float
    slope_b: float
    u_center: float
    image_width: int
    image_height: int

    def slope_at(self, v: float) -> float:
        """Berechnet die Scherungssteigung bei der gegebenen Y-Position v."""
        return self.slope_a * v + self.slope_b

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RectificationParams":
        return cls(**d)

    def __str__(self) -> str:
        return (
            f"RectificationParams("
            f"slope_a={self.slope_a:.6f}, slope_b={self.slope_b:.6f}, "
            f"u_center={self.u_center:.1f}, "
            f"image={self.image_width}×{self.image_height}px)"
        )


# ---------------------------------------------------------------------------
# JSON-I/O
# ---------------------------------------------------------------------------

def save_rectification_params(params: RectificationParams, path: str | Path) -> None:
    """Speichert Kalibrierungsparameter als JSON-Datei."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[Rectify] Kalibrierungsparameter gespeichert: {path}")


def load_rectification_params(path: str | Path) -> RectificationParams:
    """Lädt Kalibrierungsparameter aus einer JSON-Datei."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Kalibrierungsdatei nicht gefunden: {path}")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    params = RectificationParams.from_dict(d)
    print(f"[Rectify] Kalibrierungsparameter geladen: {params}")
    return params


# ---------------------------------------------------------------------------
# Bildkorrektur
# ---------------------------------------------------------------------------

def rectify_image(
    img: np.ndarray,
    params: RectificationParams,
    v_center: Optional[float] = None,
) -> np.ndarray:
    """
    Korrigiert die Y-Scherung im Bild – positionsabhängig (nicht-linear).

    Für jeden Zielpixel (dst_x, dst_y) wird der Quellpixel berechnet als:
        src_x = dst_x
        src_y = dst_y + slope(dst_y) * (dst_x - u_center)

    Da slope(v) = slope_a * v + slope_b linear von der Y-Position abhängt,
    variiert die Korrektur zeilenweise. Eine einzelne affine Matrix (warpAffine)
    könnte nur einen festen slope anwenden und wäre für Zeilen weit von
    v_center entfernt fehlerhaft. cv2.remap() löst dies exakt.

    Parameters
    ----------
    img : np.ndarray
        Graustufenbild (H×W), uint8.
    params : RectificationParams
        Kalibrierungsparameter (aus JSON geladen).
    v_center : float | None
        Nicht mehr verwendet (wird ignoriert, nur für Rückwärtskompatibilität
        der Signatur erhalten). Korrektur ist jetzt immer positionsabhängig.

    Returns
    -------
    np.ndarray
        Rektifiziertes Bild (gleiche Größe wie Eingabe), uint8.
    """
    h, w = img.shape[:2]

    # Gitter der Zielpixel: dst_x_grid[v, u] = u, dst_y_grid[v, u] = v
    dst_x_grid, dst_y_grid = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    # Positionsabhängiger slope für jede Zeile:
    # slope_grid[v, u] = slope_a * v + slope_b  (unabhängig von u)
    slope_grid = (params.slope_a * dst_y_grid + params.slope_b).astype(np.float32)

    # Quellkoordinaten (inverse map): woher kommt der Wert für Zielpixel (u, v)?
    src_x = dst_x_grid                                              # keine X-Verschiebung
    src_y = dst_y_grid + slope_grid * (dst_x_grid - params.u_center)  # Y-Verschiebung

    return cv2.remap(
        img, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=128,
    )


# ---------------------------------------------------------------------------
# Steigungsmessung
# ---------------------------------------------------------------------------

def measure_line_slope(
    img: np.ndarray,
    y_approx: float,
    search_radius: int = 8,
) -> tuple[float, float]:
    """
    Misst die Steigung einer hellen/dunklen Linie bei einer bestimmten Y-Position.

    Algorithmus: Gewichteter Schwerpunkt (Intensitäts-Schwerpunkt) pro Bildspalte,
    danach linearer Fit der Schwerpunkt-Y-Koordinaten über alle Spalten.

    Parameters
    ----------
    img : np.ndarray
        Graustufenbild (H×W), uint8.
    y_approx : float
        Ungefähre Y-Position der Linie (Pixel).
    search_radius : int
        Suchbereich: ±search_radius Zeilen um y_approx.

    Returns
    -------
    slope : float
        Steigung der Linie (dy/dx in Pixeln/Pixel).
    y_center : float
        Mittlere Y-Position der Linie (Pixel).
    """
    h, w = img.shape[:2]
    y0 = max(0, int(y_approx) - search_radius)
    y1 = min(h, int(y_approx) + search_radius + 1)
    strip = img[y0:y1, :].astype(np.float32)

    y_coords = []
    for u in range(w):
        col = strip[:, u]
        total = col.sum()
        if total > 1e-6:
            y_local = float(np.sum(col * np.arange(len(col))) / total)
        else:
            y_local = float(y1 - y0) / 2.0
        y_coords.append(y0 + y_local)

    x_vals = np.arange(w, dtype=float)
    coeffs = np.polyfit(x_vals, y_coords, 1)
    slope = float(coeffs[0])
    y_center = float(np.mean(y_coords))
    return slope, y_center


# ---------------------------------------------------------------------------
# RGB-Linien-Erkennung
# ---------------------------------------------------------------------------

def detect_colored_lines(
    img_bgr: np.ndarray,
    hue_center: float,
    hue_tol: float = 15.0,
    min_width_px: int = 20,
) -> list[tuple[float, float]]:
    """
    Erkennt farbige Linien im Bild per HSV-Hue-Filter und misst deren Steigungen.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR-Farbbild, uint8.
    hue_center : float
        Ziel-Hue-Wert in OpenCV-HSV (0–180).
        Rot ≈ 0 oder 180, Grün ≈ 60, Blau ≈ 120.
    hue_tol : float
        Toleranz um den Hue-Wert (±hue_tol).
    min_width_px : int
        Minimale Breite einer erkannten Linie (filtert Rauschen heraus).

    Returns
    -------
    list of (slope, y_center)
        Steigungen und Y-Mittelpositionen, sortiert nach y_center.
    """
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h_lo = max(0.0, hue_center - hue_tol)
    h_hi = min(180.0, hue_center + hue_tol)
    mask = cv2.inRange(hsv,
                       np.array([h_lo, 50, 50], dtype=np.uint8),
                       np.array([h_hi, 255, 255], dtype=np.uint8))

    # Sonderfall Rot: Hue-Wrap um 180
    if hue_center < hue_tol:
        mask2 = cv2.inRange(hsv,
                            np.array([180.0 - (hue_tol - hue_center), 50, 50], dtype=np.uint8),
                            np.array([180, 255, 255], dtype=np.uint8))
        mask = cv2.bitwise_or(mask, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < min_width_px:
            continue
        y_mid = float(by + bh / 2.0)
        radius = max(bh // 2, 4, 6)
        slope, y_center = measure_line_slope(gray, y_mid, search_radius=radius)
        results.append((slope, y_center))

    results.sort(key=lambda s: s[1])
    return results


# ---------------------------------------------------------------------------
# Modell-Fit
# ---------------------------------------------------------------------------

def fit_slope_model(
    slopes_and_y: list[tuple[float, float]],
) -> tuple[float, float]:
    """
    Fittet das lineare Modell slope(v) = slope_a * v + slope_b.

    Parameters
    ----------
    slopes_and_y : list of (slope, y_center)
        Gemessene Steigungen mit zugehörigen Y-Positionen (mindestens 2).

    Returns
    -------
    slope_a, slope_b : float
    """
    if len(slopes_and_y) < 2:
        raise ValueError(
            f"Mindestens 2 Messpunkte für den Modell-Fit benötigt, "
            f"aber nur {len(slopes_and_y)} vorhanden."
        )
    y_vals = np.array([s[1] for s in slopes_and_y], dtype=float)
    s_vals = np.array([s[0] for s in slopes_and_y], dtype=float)
    coeffs = np.polyfit(y_vals, s_vals, 1)
    slope_a = float(coeffs[0])
    slope_b = float(coeffs[1])

    predicted = np.polyval(coeffs, y_vals)
    residuals = s_vals - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    print(f"[Rectify] Fit: slope(v) = {slope_a:.6f}*v + {slope_b:.6f}  RMSE={rmse:.6f}")
    return slope_a, slope_b


# ---------------------------------------------------------------------------
# Schnelltest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Verwendung: python rectify.py bild.tif [y_approx]")
        sys.exit(1)

    from image_utils import load_and_crop
    img = load_and_crop(sys.argv[1])

    if len(sys.argv) >= 3:
        y_approx = float(sys.argv[2])
        slope, y_ctr = measure_line_slope(img, y_approx)
        print(f"Gemessene Steigung bei y≈{y_approx:.1f}: slope={slope:.5f}, y_center={y_ctr:.1f}")
    else:
        print(f"Bild geladen: {img.shape[1]}×{img.shape[0]} px")
