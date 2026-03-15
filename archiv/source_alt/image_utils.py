"""
Bildlade- und Vorverarbeitungs-Hilfsfunktionen für alle Datensätze.

Versuchsreihe 2026-02-26 (diese Datei):
    Zwei Kameramodi:
      - Versuch 1-6  : 128×1000 px, 2000 fps
      - Versuch 8-11 : 128×880  px, 6000 fps
    Kamerawinkel: seitlicher Versatz 500 mm, Normale 330 mm → ca. 56,6°
    Markierungsabstand: 20 mm (beide Modi)
    Format: direkt verwendbar, kein Crop/Rotate nötig.

Vorgänger-Messreihen (Rückwärtskompatibilität):
    Neue Messreihe 2026-02-13: 1000×284 px Roh-Frame → Crop + 90° → 768×72 px
    Neue Versuchsreihe 2026-02-23: 128×1000 px, schräge Kamera, rectify.py-Korrektur
    Alte Messreihe 2021: 384×232 px, kein Crop

Die Funktion load_and_crop() erkennt das Format automatisch anhand der Bildgröße.
load_and_crop_calibrated() wendet zusätzlich die Kamerakalibrierung (camera_calibration.json)
oder die Scherungskorrektur (rectification_params.json) an.
"""

import json
import warnings
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Kamera-ROI Konstanten – Messreihe 2026-02-13 (Vollbild mit Metadaten-Streifen)
# ---------------------------------------------------------------------------

FULL_H = 284
FULL_W = 1000

META_STRIP_W = 232
ROI_ROW_START = 107
ROI_ROW_END   = 179
ROI_COL_START = 232
ROI_COL_END   = 1000
ROI_H = ROI_ROW_END - ROI_ROW_START   # 72
ROI_W = ROI_COL_END - ROI_COL_START   # 768

# MATLAB-vorverarbeitete Bilder (2026-02-13)
ROI_PREPROCESSED_H = 768
ROI_PREPROCESSED_W = 72

# Alte Messreihe 2021
OLD_H = 232
OLD_W = 384

# ---------------------------------------------------------------------------
# Kamera-Konstanten – Versuchsreihe 2026-02-23 & 2026-02-26
# ---------------------------------------------------------------------------

# Versuchsreihe 2026-02-23 und 2026-02-26 Versuch 1-6 (2000 fps)
NEW_V3_H = 1000
NEW_V3_W = 128

# Versuchsreihe 2026-02-26 Versuch 8-11 (6000 fps)
NEW_V3_H2 = 880
NEW_V3_W2 = 128


# ---------------------------------------------------------------------------
# Laden mit Unicode-Pfad-Support (Windows + Sonderzeichen)
# ---------------------------------------------------------------------------

def imread_gray(path: str) -> np.ndarray:
    """
    Lädt ein Graustufenbild robust, auch bei Nicht-ASCII-Pfaden (Windows ß, ä, ö usw.).
    Gibt uint8-Array zurück.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {path}")
    return img


# ---------------------------------------------------------------------------
# Crop: Format-Erkennung anhand Bildgröße
# ---------------------------------------------------------------------------

def crop_to_roi(img: np.ndarray) -> np.ndarray:
    """
    Extrahiert das eigentliche Fadenbild aus dem vollen Kamera-Frame.

    Unterstützte Formate:
      1000×284   → Neue Messreihe 2026-02-13: Crop + 90° CW → 768×72 px
      768×72     → MATLAB-vorverarbeitet: direkt verwenden
      384×232    → Alte Messreihe 2021: direkt verwenden
      128×1000   → Versuchsreihe 2026-02-23 / 2026-02-26 Versuch 1-6: direkt verwenden
      128×880    → Versuchsreihe 2026-02-26 Versuch 8-11: direkt verwenden
      Andere     → Warnung, unverändert zurückgeben

    Parameters
    ----------
    img : np.ndarray
        Graustufenbild (H×W), uint8

    Returns
    -------
    np.ndarray
        Gecroptes Fadenbild, uint8
    """
    h, w = img.shape[:2]

    if h == FULL_H and w == FULL_W:
        # Neue Messreihe 2026-02-13: Crop + 90° Drehung im Uhrzeigersinn
        roi = img[ROI_ROW_START:ROI_ROW_END, ROI_COL_START:ROI_COL_END]
        return np.rot90(roi, k=-1)   # k=-1 = 90° CW → 768×72 px

    elif h == ROI_PREPROCESSED_H and w == ROI_PREPROCESSED_W:
        # MATLAB-vorverarbeitete Bilder: bereits 768×72 px
        return img

    elif h == OLD_H and w == OLD_W:
        # Alte Messreihe 2021 – kein Crop nötig
        return img

    elif h == NEW_V3_H and w == NEW_V3_W:
        # Versuchsreihe 2026-02-23 / 2026-02-26 Versuch 1-6 (128×1000 px)
        return img

    elif h == NEW_V3_H2 and w == NEW_V3_W2:
        # Versuchsreihe 2026-02-26 Versuch 8-11 (128×880 px, 6000 fps)
        return img

    else:
        warnings.warn(
            f"Unbekannte Bildgröße {w}×{h} px. Kein Crop angewendet. "
            f"Erwartet: {FULL_W}×{FULL_H}, {ROI_PREPROCESSED_W}×{ROI_PREPROCESSED_H}, "
            f"{OLD_W}×{OLD_H}, {NEW_V3_W}×{NEW_V3_H} oder {NEW_V3_W2}×{NEW_V3_H2}."
        )
        return img


def load_and_crop(path: str) -> np.ndarray:
    """
    Lädt ein Bild und gibt das gecropte Fadenbild zurück.
    Kombiniert imread_gray() + crop_to_roi().
    Empfohlene Basisfunktion für alle Skripte.
    """
    img = imread_gray(path)
    return crop_to_roi(img)


# ---------------------------------------------------------------------------
# Kalibrierungs-Hilfsfunktionen
# ---------------------------------------------------------------------------

def load_camera_calibration(path: str) -> dict:
    """
    Lädt camera_calibration.json (erstellt von 1_calibrate_camera.py).

    Returns
    -------
    dict mit Schlüsseln: shear_model, scale_v_model, scale_h_model,
                         homography, image_width, image_height, u_center, …
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Kalibrierungsdatei nicht gefunden: {path}")
    with open(path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    print(f"[ImageUtils] Kamerakalibrierung geladen: {path.name}")
    return calib


def load_and_crop_rectified(path: str, rectify_params=None) -> np.ndarray:
    """
    Lädt, croppt und rektifiziert optional ein Bild.
    Rückwärtskompatibel mit dem --rectify Workflow (rectification_params.json).

    Parameters
    ----------
    path : str
        Pfad zur Bilddatei.
    rectify_params : RectificationParams | None
        Wenn angegeben, wird Y-Scherung korrigiert.

    Returns
    -------
    np.ndarray : Gecroptes (und ggf. rektifiziertes) Graustufenbild, uint8.
    """
    img = load_and_crop(path)
    if rectify_params is not None:
        from rectify import rectify_image
        img = rectify_image(img, rectify_params)
    return img


def load_and_crop_calibrated(
    path: str,
    calib_params: dict | None = None,
    rectify_params=None,
) -> np.ndarray:
    """
    Lädt, croppt und wendet Kamerakalibrierung / Scherungskorrektur an.

    Priorität der Korrekturen:
      1. calib_params (camera_calibration.json, von 1_calibrate_camera.py)
      2. rectify_params (rectification_params.json, älterer Workflow)
      3. Kein Korrektur (passthrough)

    Parameters
    ----------
    path : str
        Pfad zur Bilddatei.
    calib_params : dict | None
        Ausgabe von load_camera_calibration(). Enthält shear_model mit slope_a/slope_b.
    rectify_params : RectificationParams | None
        Alternativ: RectificationParams-Objekt (älterer Workflow).

    Returns
    -------
    np.ndarray : Gecroptes und korrigiertes Graustufenbild, uint8.
    """
    img = load_and_crop(path)

    if calib_params is not None:
        # Scherungskorrektur aus camera_calibration.json extrahieren
        from rectify import rectify_image, RectificationParams
        shear = calib_params["shear_model"]
        h, w = img.shape[:2]
        u_ctr = calib_params.get("u_center", w / 2.0)
        rp = RectificationParams(
            slope_a=shear["slope_a"],
            slope_b=shear["slope_b"],
            u_center=u_ctr,
            image_width=w,
            image_height=h,
        )
        img = rectify_image(img, rp)

    elif rectify_params is not None:
        from rectify import rectify_image
        img = rectify_image(img, rectify_params)

    return img


# ---------------------------------------------------------------------------
# Pixelmaßstab-Kalibrierung (Hilfsfunktionen, rückwärtskompatibel)
# ---------------------------------------------------------------------------

def compute_pixel_size_mm(
    point1_px: tuple[float, float],
    point2_px: tuple[float, float],
    known_distance_mm: float,
) -> float:
    """
    Berechnet den Pixelmaßstab aus zwei Punkten bekannter physikalischer Distanz.

    Parameters
    ----------
    point1_px, point2_px : (x, y) in Pixeln
    known_distance_mm    : Physikalischer Abstand zwischen den Punkten (mm)

    Returns
    -------
    float : Pixelgröße in mm/px
    """
    dist_px = np.sqrt(
        (point2_px[0] - point1_px[0])**2 +
        (point2_px[1] - point1_px[1])**2
    )
    if dist_px < 1e-6:
        raise ValueError("Die beiden Punkte sind identisch.")
    return known_distance_mm / dist_px


def pixel_size_mm_linear(
    y_center_px: float,
    fit_a: float,
    fit_b: float,
) -> float:
    """
    Positionsabhängiger Pixelmaßstab (linearer Fit über Y-Position).
    fit_a = Steigung (mm/px²), fit_b = Intercept (mm/px).
    Wenn fit_a == 0: fester Maßstab = fit_b mm/px.
    """
    return fit_a * y_center_px + fit_b


def px_per_mm_at(v: float, coeffs: list, degree: int = 2) -> float:
    """
    Liefert den Pixelmaßstab [px/mm] an Y-Position v.
    Auswertet das Polynom aus camera_calibration.json (scale_v_model oder scale_h_model).

    Parameters
    ----------
    v      : Y-Position im Bild (Pixel)
    coeffs : Polynomkoeffizienten [a_n, …, a_1, a_0] (höchste Potenz zuerst)
    degree : Polynomgrad (zur Dokumentation, wird nicht aktiv genutzt)

    Returns
    -------
    float : Maßstab in px/mm an Position v
    """
    return float(np.polyval(coeffs, v))


# ---------------------------------------------------------------------------
# Maßstab-Bild laden
# ---------------------------------------------------------------------------

def load_scale_image(path: str) -> np.ndarray:
    """Lädt ein Maßstab-Bild und croppt es auf den Fadenbild-Bereich."""
    return load_and_crop(path)


# ---------------------------------------------------------------------------
# Schnelltest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Verwendung: python image_utils.py /pfad/zu/bild.tif")
        sys.exit(1)

    path = sys.argv[1]
    img_full = imread_gray(path)
    img_roi  = crop_to_roi(img_full)

    print(f"Voller Frame:  {img_full.shape[1]}×{img_full.shape[0]} px")
    print(f"Gecropter ROI: {img_roi.shape[1]}×{img_roi.shape[0]} px")
    print(f"ROI min/max:   {img_roi.min()} / {img_roi.max()}")
