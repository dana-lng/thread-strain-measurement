"""
Kamerakalibrierung aus Schachbrettmuster-Bild.

Verwendet ein Bild des Schachbrettmusters (Massstab/Checkerboard_Pattern_10mm.tif),
das mit der tatsächlichen Versuchskamera aufgenommen wurde, um:

  1. Scherungskorrektur (Y-Shear): slope(v) = slope_a * v + slope_b
     → gleiche Parameter wie rectify.py / rectification_params.json
  2. Pixelmaßstab als Polynomfunktion: mm_per_px(v) = a*v² + b*v + c
     → getrennt für vertikale (scale_v) und horizontale (scale_h) Richtung
  3. Homographie H: direkte Pixel(u,v) → Welt(x_mm, y_mm) Transformation

Verwendung:
    python 1_calibrate_camera.py
        --image  "../Massstab/Checkerboard_Pattern_10mm.tif"
        --square_mm 10
        --output camera_calibration.json
        [--inner_cols 2]     # wird auto-detektiert wenn nicht angegeben
        [--inner_rows 24]
        [--verify]           # zeigt Visualisierung und prüft Reproduzierbarkeit

Ausgabe (camera_calibration.json):
    {
      "square_mm": 10.0,
      "inner_cols": <auto-detektiert>,
      "inner_rows": <auto-detektiert>,
      "image_width": 128,
      "image_height": 1000,
      "u_center": 64.0,
      "homography": [[...], [...], [...]],
      "reproj_error_mean_px": 0.12,
      "reproj_error_max_px": 0.28,
      "shear_model":   {"slope_a": -5.6e-4, "slope_b": 0.285, "rmse": 0.002, "n_points": 24},
      "scale_v_model": {"degree": 2, "coeffs": [a, b, c], "rmse": 0.003, "n_points": 24,
                        "mean_mm_per_px": 0.178},
      "scale_h_model": {"degree": 2, "coeffs": [a, b, c], "rmse": 0.005, "n_points":  3,
                        "mean_mm_per_px": 0.252}
    }

Öffentliche API (importierbar von anderen Skripten):
    load_camera_calibration(path)  -> dict
    pixel_to_mm(u, v, H)           -> (x_mm, y_mm)
    mm_to_pixel(x_mm, y_mm, H)    -> (u, v)
    px_per_mm_at(v, coeffs)       -> float  [px/mm = 1/(mm/px)]

Hintergrund – Kamera-Setup (2026-02-26):
    Seitlicher Versatz: 500 mm, Normale: 330 mm → Winkel ≈ 56,6°
    Bildformat: 128×1000 px (Versuch 1-6, 2000 fps)
                128×880  px (Versuch 8-11, 6000 fps)
    Schachbrettmuster: 10 mm Kantenlänge, 8-bit Graustufen, 128×1000 px

Hinweis – Warum kein Lens-Distortion-Modell?
    Die Hochgeschwindigkeitskamera hat eine sehr kleine Apertur und ein relativ
    großes Pixel (16 µm). Radialer Linsenversatz ist bei diesem Kameratyp
    vernachlässigbar. Der dominante geometrische Effekt ist die Y-Scherung
    durch den schrägen Aufnahmewinkel, die linear über die Y-Achse variiert.
    Das Modell slope(v) = slope_a*v + slope_b erfasst diesen Effekt vollständig.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Öffentliche API (importierbar von dataset.py, infer.py, annotate.py)
# ---------------------------------------------------------------------------

def load_camera_calibration(path: str) -> dict:
    """
    Lädt camera_calibration.json (erstellt von diesem Skript).

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
    print(f"[Calib] Geladen: {path.name}  "
          f"({calib.get('inner_cols','?')}×{calib.get('inner_rows','?')} Ecken, "
          f"Reproj-Fehler ∅={calib.get('reproj_error_mean_px', float('nan')):.3f} px)")
    return calib


def pixel_to_mm(u: float, v: float, H: list) -> tuple[float, float]:
    """
    Wandelt Bildpixel (u, v) in physikalische Koordinaten (x_mm, y_mm) um.

    Parameters
    ----------
    u, v : float  –  Bildkoordinaten in Pixeln.
    H    : list   –  3×3 Homographiematrix aus camera_calibration.json.

    Returns
    -------
    (x_mm, y_mm) : physikalische Koordinaten in Millimeter.
    """
    H_np = np.array(H, dtype=np.float64)
    p_h = H_np @ np.array([u, v, 1.0])
    return float(p_h[0] / p_h[2]), float(p_h[1] / p_h[2])


def mm_to_pixel(x_mm: float, y_mm: float, H: list) -> tuple[float, float]:
    """
    Inverse Homographie: physikalische Koordinate (x_mm, y_mm) → Bildpixel (u, v).
    H ist die Vorwärts-Homographie (Pixel → Welt); sie wird intern invertiert.
    """
    H_inv = np.linalg.inv(np.array(H, dtype=np.float64))
    p_h = H_inv @ np.array([x_mm, y_mm, 1.0])
    return float(p_h[0] / p_h[2]), float(p_h[1] / p_h[2])


def px_per_mm_at(v: float, coeffs: list, degree: int = 2) -> float:
    """
    Liefert den Maßstab [px/mm] an Y-Position v (z.B. für Dehnungsberechnung).

    Parameters
    ----------
    v      : Y-Position im Bild (Pixel).
    coeffs : Polynomkoeffizienten [a_n, …, a_1, a_0] aus scale_v_model.
             Einheit des Polynoms: mm/px. Kehrwert = px/mm.

    Returns
    -------
    float : Maßstab in px/mm = Pixel pro Millimeter.
    """
    mm_per_px = float(np.polyval(coeffs, v))
    return 1.0 / mm_per_px if mm_per_px > 1e-9 else float("inf")


# ---------------------------------------------------------------------------
# Schachbrettecken erkennen
# ---------------------------------------------------------------------------

def _corners_via_profile(
    img_eq: np.ndarray,
    inner_cols: int,
    inner_rows: int,
) -> np.ndarray:
    """
    Sobel-Kantenprojektions-basierte Eckenerkennung – funktioniert für JEDE Mustergröße,
    auch für schmale Muster mit inner_cols < 3 (die cv2.findChessboardCorners ablehnt).

    Algorithmus:
      1. Sobel-Kantenbetrag in X- und Y-Richtung berechnen
      2. Vertikale Kanten auf X-Achse projizieren → Peaks = Spalten-Übergänge
      3. Horizontale Kanten auf Y-Achse projizieren → Peaks = Reihen-Übergänge
      4. Kreuzungspunkte der Gitterlinien = Ecken
      5. Sub-Pixel-Verfeinerung via cv2.cornerSubPix

    Warum Sobel statt Intensitätsprofil:
      Ein Schachbrettmuster hat pro Spalte/Zeile annähernd gleich viele helle und
      dunkle Pixel → mittlere Intensität ≈ konstant → Intensitätsprofil ist flach,
      enthält keine Übergangsinformation. Sobel-Kantenbetrag hat dagegen klare
      Maxima an den Feldübergängen.

    Returns
    -------
    np.ndarray [inner_rows * inner_cols, 1, 2] float32
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    h, w = img_eq.shape[:2]

    # Sobel-Kantenbetrag: stark an Schwarz-Weiß-Übergängen des Schachbretts
    sobel_x = cv2.Sobel(img_eq.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_eq.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    # Projektion auf Achsen: Mittelwert über senkrechte Dimension
    x_profile = np.abs(sobel_x).mean(axis=0)   # (w,) – Peaks bei Feldspalten-Grenzen
    y_profile = np.abs(sobel_y).mean(axis=1)   # (h,) – Peaks bei Feldreihen-Grenzen

    def find_inner_lines(profile: np.ndarray, n_inner: int) -> np.ndarray:
        """
        Findet n_inner innere Gitterlinien in einem 1D-Kantenprojektionsprofil.

        Das Schachbrett hat n_inner+1 Felder und damit n_inner interne Grenzen.
        Peaks im Sobel-Profil liegen direkt an den Feldübergängen.
        """
        n = len(profile)
        field_size_est = n / (n_inner + 1)

        # Gauß-Glättung: entfernt Hochfrequenzrauschen, behält Feldgrenzen-Peaks
        sigma = max(1.5, field_size_est * 0.15)
        smooth = gaussian_filter1d(profile.astype(float), sigma=sigma)

        # Peaks direkt im Sobel-Profil (keine Ableitung nötig – Sobel liefert bereits Kantenmaximum)
        min_dist = max(3, int(field_size_est * 0.30))
        peaks, _ = find_peaks(smooth, distance=min_dist)

        if len(peaks) == 0:
            # Kein Peak → gleichabständige Fallback-Positionen
            return np.linspace(field_size_est, n - field_size_est, n_inner)

        # Peaks nahe am Bildrand ausschließen (< 0.6 × Feldbreite vom Rand):
        # Randkanten (Außenrand des Schachbretts oder Bildgrenze) erzeugen starke Sobel-Peaks,
        # sind aber keine inneren Ecken. Innere Ecken liegen typisch ≥ 1 Feldbreite vom Rand.
        # 0.6 statt 0.5: schließt x≈17 (Schachbrett-Außenrand) aus, behält x≈33 (1. innere Ecke).
        margin = field_size_est * 0.6
        peaks = peaks[(peaks >= margin) & (peaks <= n - margin)]
        if len(peaks) == 0:
            return np.linspace(field_size_est, n - field_size_est, n_inner)

        # Zu viele Peaks: n_inner prominenteste (höchste Amplitude) auswählen
        if len(peaks) > n_inner:
            top_idx = np.argsort(smooth[peaks])[::-1][:n_inner]
            peaks = np.sort(peaks[top_idx])

        # Zu wenige Peaks: gleichabständige Positionen als Fallback
        if len(peaks) < n_inner:
            print(f"[Calib]   Kantenprojekt.: nur {len(peaks)}/{n_inner} Peaks gefunden, "
                  f"fülle mit gleichabständigen auf.")
            peaks = np.linspace(field_size_est, n - field_size_est, n_inner)

        return peaks.astype(float)

    inner_x = find_inner_lines(x_profile, inner_cols)
    inner_y = find_inner_lines(y_profile, inner_rows)

    print(f"[Calib] Kantenprojektions-Methode: {len(inner_x)} Spalten-Linien × "
          f"{len(inner_y)} Reihen-Linien gefunden")

    # Ecken in Zeilen-Hauptreihenfolge (wie cv2.findChessboardCorners)
    corners = np.array(
        [[[float(x), float(y)]] for y in inner_y for x in inner_x],
        dtype=np.float32,
    )

    # Sub-Pixel-Verfeinerung: Fenster (11,11) = ±11 px Suchradius.
    # Größer als Standard (7,7), damit bei scherungsbedingten X-Abweichungen
    # (Sattelpunkt bis ~10 px vom Profilschätzwert entfernt) sicher konvergiert wird.
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    try:
        corners = cv2.cornerSubPix(img_eq, corners, (11, 11), (-1, -1), subpix_criteria)
    except cv2.error:
        pass  # Selten; grobe Positionen bleiben dann erhalten

    return corners


def detect_corners(
    img: np.ndarray,
    inner_cols: int | None,
    inner_rows: int | None,
) -> tuple[np.ndarray, int, int]:
    """
    Erkennt Schachbrettecken mit Sub-Pixel-Genauigkeit.

    Strategie (in dieser Reihenfolge):
      1. cv2.findChessboardCorners (schnell, präzise) – nur wenn beide Dimensionen ≥ 3
      2. Profil-basierte Methode (_corners_via_profile) – für schmale Muster (inner_cols < 3)
         und als Fallback wenn OpenCV das Muster nicht findet.

    Hintergrund: OpenCV erfordert inner_cols ≥ 3 AND inner_rows ≥ 3.
    Das Schachbrettmuster dieser Versuchsreihe (128px breit, ~3 Felder × ~25 Felder)
    hat nur inner_cols ≈ 2, deshalb ist der Profil-Fallback zwingend erforderlich.

    Parameters
    ----------
    img        : Graustufenbild (uint8), nach crop_to_roi
    inner_cols : Anzahl innerer Ecken in X-Richtung (None = auto-geschätzt)
    inner_rows : Anzahl innerer Ecken in Y-Richtung (None = auto-geschätzt)

    Returns
    -------
    corners   : np.ndarray [N, 1, 2]  Sub-Pixel-Ecken (float32)
    used_cols : tatsächlich verwendete Spaltenanzahl
    used_rows : tatsächlich verwendete Reihenanzahl
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    cv2_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FAST_CHECK)

    h, w = img.shape[:2]

    # ---- Fall 1: Explizite Angabe ----
    if inner_cols is not None and inner_rows is not None:

        if inner_cols >= 3 and inner_rows >= 3:
            # Versuch 1: findChessboardCornersSB (Sattelpunkt-basiert).
            # Findet direkt die 2×2-alternierenden Sattelpunkte ohne Intensitätsschwellwert.
            # Robuster als findChessboardCorners bei Scherung, kleinen Mustern und
            # geringem Kontrast. Integriert bereits Sub-Pixel-Genauigkeit (CALIB_CB_ACCURACY).
            flags_sb = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE |
                        cv2.CALIB_CB_ACCURACY)
            for src in (img_eq, img):
                found, corners = cv2.findChessboardCornersSB(
                    src, (inner_cols, inner_rows), flags_sb)
                if found:
                    print(f"[Calib] findChessboardCornersSB: {inner_cols}×{inner_rows} "
                          f"Ecken erkannt (Sattelpunkt-basiert)")
                    return corners, inner_cols, inner_rows

            # Versuch 2: findChessboardCorners + cornerSubPix (Fenster 11×11)
            for src in (img_eq, img):
                found, corners = cv2.findChessboardCorners(src, (inner_cols, inner_rows),
                                                           cv2_flags)
                if found:
                    corners = cv2.cornerSubPix(img_eq, corners,
                                               (11, 11), (-1, -1), subpix_criteria)
                    print(f"[Calib] findChessboardCorners: {inner_cols}×{inner_rows} Ecken erkannt")
                    return corners, inner_cols, inner_rows

        # Fallback: Profil-Methode (für inner_cols < 3 oder wenn OpenCV versagt)
        print(f"[Calib] Verwende Profil-Methode für {inner_cols}×{inner_rows} Muster "
              f"({'OpenCV benötigt ≥3 in jeder Richtung' if inner_cols < 3 or inner_rows < 3 else 'OpenCV-Fallback'})")
        corners = _corners_via_profile(img_eq, inner_cols, inner_rows)
        return corners, inner_cols, inner_rows

    # ---- Fall 2: Auto-Detektion ----
    # Schätze innere Ecken aus der Bildgröße:
    # Typisch für 128×1000 px mit 10mm-Feldern (~0.2mm/px):
    #   X: 128 * 0.2 / 10 ≈ 2.5 Felder → inner_cols ≈ 2
    #   Y: 1000 * 0.2 / 10 ≈ 20 Felder → inner_rows ≈ 19

    # Schritt A: Versuche OpenCV mit Dimensionen ≥ 3
    print("[Calib] Auto-Detektion: versuche findChessboardCornersSB + findChessboardCorners…")
    flags_sb = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    for nc in range(3, 8):
        for nr in range(3, h // 8):
            found, corners = cv2.findChessboardCornersSB(img_eq, (nc, nr), flags_sb)
            if found:
                print(f"[Calib] findChessboardCornersSB auto-erkannt: {nc}×{nr} innere Ecken")
                return corners, nc, nr
            found, corners = cv2.findChessboardCorners(img_eq, (nc, nr), cv2_flags)
            if found:
                corners = cv2.cornerSubPix(img_eq, corners,
                                           (11, 11), (-1, -1), subpix_criteria)
                print(f"[Calib] findChessboardCorners auto-erkannt: {nc}×{nr} innere Ecken")
                return corners, nc, nr

    # Schritt B: Schätze innere Ecken aus Sobel-Kantenprojektionen und nutze Profil-Methode
    print("[Calib] OpenCV-Auto-Detektion fehlgeschlagen. Schätze Muster aus Sobel-Kanten…")
    # Sobel ist deutlich zuverlässiger als Intensitätsprofil für Schachbrettmuster:
    # Intensitätsprofil ≈ konstant (je Spalte/Zeile ~50% hell/dunkel), Sobel hat Peaks an Feldübergängen.
    from scipy.signal import find_peaks
    sobel_x_est = cv2.Sobel(img_eq.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y_est = cv2.Sobel(img_eq.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    x_profile_est = np.abs(sobel_x_est).mean(axis=0)
    y_profile_est = np.abs(sobel_y_est).mean(axis=1)

    x_peaks, _ = find_peaks(x_profile_est, distance=max(3, w // 8))
    nc_est = max(1, min(len(x_peaks), 6))  # Sobel-Peaks = Feldübergänge = innere Spaltenecken

    y_peaks, _ = find_peaks(y_profile_est, distance=max(3, h // max(nc_est * 15, 3)))
    nr_est = max(3, min(len(y_peaks), h // 5))

    print(f"[Calib] Sobel-Schätzung: inner_cols≈{nc_est}  inner_rows≈{nr_est}")
    print(f"[Calib] Tipp: Mit --inner_cols {nc_est} --inner_rows {nr_est} explizit angeben.")

    corners = _corners_via_profile(img_eq, nc_est, nr_est)
    return corners, nc_est, nr_est


# ---------------------------------------------------------------------------
# Welt-Koordinaten (Gitterebene des Schachbretts)
# ---------------------------------------------------------------------------

def build_world_points(
    inner_cols: int,
    inner_rows: int,
    square_mm: float,
) -> np.ndarray:
    """
    Erstellt 2D-Welt-Koordinaten der inneren Schachbrettecken.

    Koordinatensystem: Ursprung = erste innere Ecke oben-links.
    X wächst in Schachbrett-Spaltenrichtung, Y in Zeilenrichtung.

    Returns
    -------
    np.ndarray [N, 1, 2] float32  (Koordinaten in mm)
    """
    obj_pts = [
        [c * square_mm, r * square_mm]
        for r in range(inner_rows)
        for c in range(inner_cols)
    ]
    return np.array(obj_pts, dtype=np.float32).reshape(-1, 1, 2)


# ---------------------------------------------------------------------------
# Homographie (Pixel → Welt-mm)
# ---------------------------------------------------------------------------

def compute_homography(
    corners_px: np.ndarray,
    world_pts: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Berechnet die Homographie-Matrix H: Bildpixel → Welt (mm).

    Die Homographie nimmt an, dass alle Schachbrettecken in einer Ebene liegen.
    Sie ist exakt genug für unsere schmale Kamera (128 px Breite), da die
    Perspektivverzerrung in X vernachlässigbar ist.

    Parameters
    ----------
    corners_px : [N, 1, 2] Bildkoordinaten (Sub-Pixel)
    world_pts  : [N, 1, 2] Weltkoordinaten in mm

    Returns
    -------
    H           : 3×3 Homographiematrix (np.float64)
    reproj_mean : mittlerer Reprojektionsfehler in Pixeln
    reproj_max  : maximaler Reprojektionsfehler in Pixeln
    """
    src = corners_px.reshape(-1, 2).astype(np.float64)
    dst = world_pts.reshape(-1, 2).astype(np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

    if H is None:
        raise RuntimeError(
            "Homographie konnte nicht berechnet werden.\n"
            "Mögliche Ursachen: Zu wenige Ecken, Ecken liegen in einer Linie."
        )

    # Reprojektionsfehler: Welt → Pixel (Rückrechnung über inverse H)
    H_inv = np.linalg.inv(H)
    errors = []
    for pt_px, pt_mm in zip(src, dst):
        p_h = H_inv @ np.array([pt_mm[0], pt_mm[1], 1.0])
        u_r = p_h[0] / p_h[2]
        v_r = p_h[1] / p_h[2]
        errors.append(float(np.sqrt((u_r - pt_px[0])**2 + (v_r - pt_px[1])**2)))

    reproj_mean = float(np.mean(errors))
    reproj_max  = float(np.max(errors))
    print(f"[Calib] Reprojektionsfehler: ∅={reproj_mean:.4f} px, max={reproj_max:.4f} px")
    return H, reproj_mean, reproj_max


# ---------------------------------------------------------------------------
# Scherungsmodell aus horizontalen Eckzeilen
# ---------------------------------------------------------------------------

def compute_shear_model(
    corners_px: np.ndarray,
    inner_cols: int,
    inner_rows: int,
) -> dict:
    """
    Berechnet das Y-Scherungsmodell slope(v) = slope_a * v + slope_b.

    Für jede horizontale Zeile des Schachbretts wird die Steigung der
    Eckpunkte (dy/dx) per linearem Fit gemessen. Die Y-Positions-Abhängigkeit
    dieser Steigungen ergibt das lineare Scherungsmodell.

    Bei nur 1 innerer Spalte ist keine Steigung messbar → slope_a=slope_b=0.

    Returns
    -------
    dict: {slope_a, slope_b, rmse [px/px], n_points}
    """
    pts = corners_px.reshape(inner_rows, inner_cols, 2)

    if inner_cols < 2:
        print("[Calib] Warnung: Nur 1 innere Spalte – Scherungsmodell aus "
              "vertikaler Neigung geschätzt.")
        # Alternativ: vertikale Spalte nutzen um Scherung zu schätzen
        # (weniger präzise, aber besser als 0)
        slopes = []
        y_ctr  = []
        for c in range(inner_cols):
            col_pts = pts[:, c, :]   # [inner_rows, 2]
            if len(col_pts) >= 4:
                xs = col_pts[:, 0].astype(float)
                ys = col_pts[:, 1].astype(float)
                # Erwartete X-Positionen sind identisch (gleiche Spalte)
                # → jede Abweichung von x=const ist Scherungseffekt
                dx = xs - xs.mean()  # Abweichung von Ideal
                if len(dx) >= 2 and np.std(ys) > 1.0:
                    coeff = np.polyfit(ys, dx, 1)
                    # Steigung: wie viel x-Verschiebung pro y-Position
                    # Das entspricht slope (dy/dx für horizontale Linien)
                    slopes.append(float(coeff[0]))
                    y_ctr.append(float(np.mean(ys)))
        if len(slopes) < 2:
            print("[Calib]   Nicht genug Messpunkte → slope_a=0, slope_b=0")
            return {"slope_a": 0.0, "slope_b": 0.0, "rmse": 0.0, "n_points": 0}
        fit = np.polyfit(np.array(y_ctr), np.array(slopes), 1)
        slope_a, slope_b = float(fit[0]), float(fit[1])
        pred = np.polyval(fit, np.array(y_ctr))
        rmse = float(np.sqrt(np.mean((np.array(slopes) - pred)**2)))
        print(f"[Calib] Scherung (1-Spalten-Schätzung): slope_a={slope_a:.6f}  slope_b={slope_b:.6f}")
        return {"slope_a": slope_a, "slope_b": slope_b, "rmse": rmse, "n_points": len(slopes)}

    # Normalfall: >= 2 innere Spalten
    slopes = []
    y_centers = []
    for r in range(inner_rows):
        row_pts = pts[r]   # [inner_cols, 2]
        xs = row_pts[:, 0].astype(float)
        ys = row_pts[:, 1].astype(float)
        coeffs = np.polyfit(xs, ys, 1)
        slopes.append(float(coeffs[0]))
        y_centers.append(float(np.mean(ys)))

    y_arr = np.array(y_centers)
    s_arr = np.array(slopes)
    fit_coeffs = np.polyfit(y_arr, s_arr, 1)
    slope_a = float(fit_coeffs[0])
    slope_b = float(fit_coeffs[1])
    rmse = float(np.sqrt(np.mean((s_arr - np.polyval(fit_coeffs, y_arr))**2)))
    print(f"[Calib] Scherungsmodell: slope(v) = {slope_a:.6f}*v + {slope_b:.6f}  "
          f"RMSE={rmse:.6f}  n={len(slopes)}")
    return {"slope_a": slope_a, "slope_b": slope_b, "rmse": rmse, "n_points": len(slopes)}


# ---------------------------------------------------------------------------
# Pixelmaßstab-Modell (mm/px als Polynomfunktion der Y-Position)
# ---------------------------------------------------------------------------

def compute_scale_model(
    corners_px: np.ndarray,
    inner_cols: int,
    inner_rows: int,
    square_mm: float,
    direction: str = "vertical",
    degree: int = 2,
) -> dict:
    """
    Berechnet das positionsabhängige Pixelmaßstab-Modell.

    Für jeden benachbarten Eckenpaar wird der physikalische Abstand (square_mm)
    durch den Pixelabstand dividiert → mm/px als Funktion der Y-Position.
    Polynomfit dieser Messwerte ergibt das Modell.

    direction = "vertical":
        Paare in Y-Richtung (gleiche Spalte, benachbarte Reihe).
        Wichtig für Dehnungsberechnung (Y-Koordinaten-Differenzen).

    direction = "horizontal":
        Paare in X-Richtung (gleiche Reihe, benachbarte Spalte).
        Nur sinnvoll wenn >= 2 innere Spalten vorhanden.

    Returns
    -------
    dict: {degree, coeffs [mm/px Polynom], rmse, n_points, mean_mm_per_px}
    """
    pts = corners_px.reshape(inner_rows, inner_cols, 2)
    scales = []
    v_positions = []

    if direction == "vertical":
        for c in range(inner_cols):
            for r in range(inner_rows - 1):
                dy = abs(float(pts[r + 1, c, 1]) - float(pts[r, c, 1]))
                if dy > 0.5:
                    scales.append(square_mm / dy)
                    v_positions.append((float(pts[r + 1, c, 1]) + float(pts[r, c, 1])) / 2.0)
    else:  # horizontal
        for r in range(inner_rows):
            for c in range(inner_cols - 1):
                dx = abs(float(pts[r, c + 1, 0]) - float(pts[r, c, 0]))
                if dx > 0.5:
                    scales.append(square_mm / dx)
                    v_positions.append(float(pts[r, c, 1]))

    if len(scales) < 2:
        print(f"[Calib] Warnung: Zu wenige Messpunkte für {direction} Maßstab. "
              f"Verwende mittleren Wert ohne Positionsabhängigkeit.")
        s_mean = float(np.mean(scales)) if scales else 0.2
        return {"degree": 0, "coeffs": [s_mean], "rmse": 0.0, "n_points": len(scales),
                "mean_mm_per_px": s_mean}

    use_degree = min(degree, len(scales) - 1)
    v_arr = np.array(v_positions)
    s_arr = np.array(scales)
    fit_coeffs = np.polyfit(v_arr, s_arr, use_degree)
    coeffs_list = [float(c) for c in fit_coeffs]
    predicted = np.polyval(fit_coeffs, v_arr)
    rmse = float(np.sqrt(np.mean((s_arr - predicted)**2)))
    s_mean = float(np.mean(s_arr))

    print(f"[Calib] Maßstab ({direction}): ∅={s_mean:.5f} mm/px  "
          f"[{min(s_arr):.5f}–{max(s_arr):.5f}]  RMSE={rmse:.6f}  n={len(scales)}")
    return {"degree": use_degree, "coeffs": coeffs_list, "rmse": rmse,
            "n_points": len(scales), "mean_mm_per_px": s_mean}


# ---------------------------------------------------------------------------
# Visualisierung
# ---------------------------------------------------------------------------

def visualize_results(
    img: np.ndarray,
    corners_px: np.ndarray,
    inner_cols: int,
    inner_rows: int,
    calib: dict,
    output_dir: Path,
):
    """Erstellt einen 4-Panel-Plot der Kalibrierungsergebnisse."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        # Panel 1: Bild mit erkannten Ecken
        img_col = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(img_col, (inner_cols, inner_rows), corners_px, True)
        axes[0].imshow(cv2.cvtColor(img_col, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Erkannte Ecken\n({inner_cols} Sp × {inner_rows} Reihen)")
        axes[0].set_xlabel("u (px)"); axes[0].set_ylabel("v (px)")

        # Panel 2: Reprojektionsfehler (Pixel vs. Reproj)
        src = corners_px.reshape(-1, 2).astype(float)
        pts_repr = []
        for pt in src:
            u_r, v_r = mm_to_pixel(*pixel_to_mm(pt[0], pt[1], calib["homography"]),
                                    calib["homography"])
            pts_repr.append([u_r, v_r])
        pts_repr = np.array(pts_repr)
        axes[1].scatter(src[:, 0], src[:, 1], s=12, c="blue", label="Erkannt")
        axes[1].scatter(pts_repr[:, 0], pts_repr[:, 1], s=8, c="red",
                        marker="x", label="Reprojiziert")
        axes[1].set_title(f"Reprojektionsfehler\n∅={calib['reproj_error_mean_px']:.4f} px")
        axes[1].set_xlabel("u (px)"); axes[1].set_ylabel("v (px)")
        axes[1].invert_yaxis(); axes[1].legend(fontsize=8); axes[1].set_aspect("equal")

        # Panel 3: Vertikaler Pixelmaßstab über Y
        model_v = calib["scale_v_model"]
        v_range = np.linspace(0, img.shape[0], 200)
        axes[2].plot(v_range, np.polyval(model_v["coeffs"], v_range), "r-",
                     linewidth=2, label="Polynomfit")
        axes[2].set_title("Pixelmaßstab (vertikal)")
        axes[2].set_xlabel("Y-Position (px)"); axes[2].set_ylabel("mm/px")
        axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

        # Panel 4: Scherungssteigung über Y
        shear = calib["shear_model"]
        axes[3].plot(v_range,
                     np.polyval([shear["slope_a"], shear["slope_b"]], v_range),
                     "g-", linewidth=2, label="Linearer Fit")
        axes[3].axhline(0, color="gray", linewidth=0.8, linestyle="--")
        axes[3].set_title("Y-Scherungssteigung\nslope(v) = slope_a·v + slope_b")
        axes[3].set_xlabel("Y-Position (px)"); axes[3].set_ylabel("slope [px/px]")
        axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)

        plt.suptitle(
            f"Kamerakalibrierung  –  {inner_cols}×{inner_rows} innere Ecken, "
            f"{calib['square_mm']} mm/Feld  –  {img.shape[1]}×{img.shape[0]} px",
            fontsize=10)
        plt.tight_layout()
        out = output_dir / "calibration_result.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Calib] Visualisierung gespeichert: {out}")

    except Exception as e:
        print(f"[Calib] Warnung: Visualisierung fehlgeschlagen: {e}")


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def calibrate(args) -> dict:
    """Führt die vollständige Kalibrierung durch. Gibt calib-Dict zurück."""
    from image_utils import imread_gray, crop_to_roi

    print(f"[Calib] Lade Bild: {args.image}")
    img_raw = imread_gray(args.image)
    img = crop_to_roi(img_raw)
    h, w = img.shape[:2]
    print(f"[Calib] Bildgröße: {w}×{h} px")

    corners_px, used_cols, used_rows = detect_corners(img, args.inner_cols, args.inner_rows)
    n_corners = len(corners_px)
    print(f"[Calib] {n_corners} Ecken ({used_cols} Sp × {used_rows} Reihen)")

    world_pts = build_world_points(used_cols, used_rows, args.square_mm)
    H, reproj_mean, reproj_max = compute_homography(corners_px, world_pts)
    shear_model = compute_shear_model(corners_px, used_cols, used_rows)
    scale_v = compute_scale_model(corners_px, used_cols, used_rows,
                                   args.square_mm, "vertical", degree=2)
    scale_h = compute_scale_model(corners_px, used_cols, used_rows,
                                   args.square_mm, "horizontal", degree=2)

    calib = {
        "square_mm": args.square_mm,
        "inner_cols": used_cols,
        "inner_rows": used_rows,
        "image_width": w,
        "image_height": h,
        "u_center": w / 2.0,
        "calibration_image": str(args.image),
        "corners_px": corners_px.reshape(-1, 2).tolist(),
        "world_pts_mm": world_pts.reshape(-1, 2).tolist(),
        "homography": H.tolist(),
        "reproj_error_mean_px": reproj_mean,
        "reproj_error_max_px": reproj_max,
        "shear_model": shear_model,
        "scale_v_model": scale_v,
        "scale_h_model": scale_h,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    print(f"\n[Calib] ✓ Gespeichert: {out_path}")
    print(f"         Scherung:    slope_a={shear_model['slope_a']:.6f}  "
          f"slope_b={shear_model['slope_b']:.6f}")
    print(f"         Maßstab(v): {scale_v['mean_mm_per_px']:.5f} mm/px ∅")
    print(f"         Maßstab(h): {scale_h['mean_mm_per_px']:.5f} mm/px ∅")
    print(f"         Reproj:     ∅={reproj_mean:.4f} px  max={reproj_max:.4f} px")

    if args.verify:
        visualize_results(img, corners_px, used_cols, used_rows, calib, out_path.parent)
        scale = min(900.0 / h, 400.0 / w, 1.0)
        img_show = cv2.resize(
            cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR),
            (int(w * scale), int(h * scale))
        )
        cv2.drawChessboardCorners(img_show, (used_cols, used_rows),
                                  (corners_px * scale).astype(np.float32), True)
        cv2.imshow("Schachbrettecken (beliebige Taste zum Schliessen)", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return calib


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kamerakalibrierung aus Schachbrettmuster-Bild (2026-02-26)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", default="../Massstab/Checkerboard_Pattern_10mm.tif",
        help="Pfad zum Schachbrettmuster-Bild"
    )
    parser.add_argument("--square_mm", type=float, default=10.0,
                        help="Kantenlänge eines Schachfeldes in mm")
    parser.add_argument("--inner_cols", type=int, default=None,
                        help="Anzahl innerer Ecken in X-Richtung (None=auto)")
    parser.add_argument("--inner_rows", type=int, default=None,
                        help="Anzahl innerer Ecken in Y-Richtung (None=auto)")
    parser.add_argument("--output", default="camera_calibration.json",
                        help="Ausgabedatei für Kalibrierungsparameter")
    parser.add_argument("--verify", action="store_true",
                        help="Visualisierung anzeigen und als PNG speichern")
    args = parser.parse_args()

    try:
        calibrate(args)
    except RuntimeError as e:
        print(f"\nFehler: {e}", file=sys.stderr)
        sys.exit(1)
