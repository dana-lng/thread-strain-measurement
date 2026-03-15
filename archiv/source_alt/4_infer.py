"""
Inferenz-Skript: Anwenden des trainierten Modells auf eine Versuchsreihe.

Verwendung:
    python 4_infer.py
        --model   checkpoints/best_model.pt
        --images  Versuch_1
        --calib   camera_calibration.json
        --fps     2000
        --output  Versuch_1/results.json

Wichtige Argumente:
    --model           Pfad zum gespeicherten Modell (.pt aus 3_train.py)
    --images          Ordner mit .tif Bildern der Versuchsreihe
    --calib           camera_calibration.json (Scherungskorrektur + Maßstab)
    --rectify         Alternativ: rectification_params.json
    --fps             Framerate (Standard: 2000 für Versuch 1-6;
                      6000 für Versuch 8-11)
    --marker_dist_mm  Markierungsabstand im Ruhezustand in mm (Standard: 20.0)
    --output          Ausgabe-JSON (Standard: results.json)
    --visualize       3-Panel-Debug-Video erzeugen (.avi)
    --dist_min_px     Minimaler Marker-Abstand in px (Standard: 50)
    --dist_max_px     Maximaler Marker-Abstand in px (Standard: 200)

Pixel-zu-mm-Kalibrierung:
    Wenn --calib angegeben: Positionsabhängiger Maßstab aus scale_v_model Polynom.
        scale(v) = mm_per_px_at(v) = np.polyval(coeffs, v)
        Kehrwert: px_per_mm = 1 / scale(v)
    Wenn --pixel_fit_a/b angegeben: Linearer Fit (älterer Workflow).
    Wenn weder noch: Fester Maßstab (--pixel_fit_b als Einzelwert).

Ausgabe (results.json):
    Pro Frame: {frame, filename, time_s, confidence, y_top_px, y_bot_px,
                distance_mm, strain_percent}
    Frames ohne erkannten Marker: y_top_px=None, strain_percent=None.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import build_model, MarkerNet, MarkerNetSmall
from image_utils import load_and_crop_calibrated, imread_gray


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Lädt Modell aus Checkpoint.

    Returns
    -------
    model, target_size (H, W), args_dict
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict   = ckpt.get("args", {})
    target_size = ckpt.get("target_size", None)  # None = nativ

    model_type = args_dict.get("model", "standard")
    dropout    = args_dict.get("dropout", 0.0)
    base_ch    = args_dict.get("base_ch", 32)   # aus Checkpoint lesen, nicht hartkodieren
    model = build_model(model_type, dropout=dropout, base_ch=base_ch)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    val_loss = ckpt.get("val_loss", float("nan"))
    val_mae  = ckpt.get("val_mae_px", float("nan"))
    print(f"[Infer] Modell: Epoche {ckpt.get('epoch','?')}  "
          f"Val Loss={val_loss:.5f}  Val MAE={val_mae:.2f} px  "
          f"target_size={target_size}")
    return model, target_size, args_dict


# ---------------------------------------------------------------------------
# Bildvorverarbeitung (identisch zu Dataset)
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray, target_size) -> torch.Tensor:
    """Normiert und konvertiert zu Tensor [1, 1, H, W]. Resize nur wenn target_size.

    Perzentil-Kontrastnormierung (1%–99%) identisch zu dataset.py → belichtungsinvariant.
    """
    if target_size is not None:
        h, w = target_size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    # Perzentil-Normierung (muss identisch zu MarkerDataset.__getitem__ sein)
    lo, hi = np.percentile(img, [1, 99])
    if hi > lo:
        img = np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255,
                      0, 255).astype(np.uint8)
    img_f = img.astype(np.float32) / 255.0
    return torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


# ---------------------------------------------------------------------------
# Positionsabhängiger Pixelmaßstab
# ---------------------------------------------------------------------------

def get_pixel_scale_fn(calib_params: dict | None, fit_a: float, fit_b: float):
    """
    Gibt eine Funktion zurück: v_px → mm/px.

    Wenn calib_params angegeben: Polynomfit aus scale_v_model (positionsabhängig).
    Wenn fit_a != 0: Linearer Fit (älterer Workflow).
    Sonst: Fester Maßstab fit_b mm/px.

    Parameters
    ----------
    calib_params : dict | None  (camera_calibration.json)
    fit_a        : Lineare Steigung mm/px² (0 = fester Maßstab)
    fit_b        : Linearer Intercept mm/px

    Returns
    -------
    scale_fn(v_center_px) -> float   Pixelmaßstab in mm/px
    """
    if calib_params is not None and "scale_v_model" in calib_params:
        coeffs = calib_params["scale_v_model"]["coeffs"]
        def scale_fn(v: float) -> float:
            val = float(np.polyval(coeffs, v))
            return max(val, 1e-6)
        print(f"[Infer] Pixelmaßstab: Polynom (Grad {len(coeffs)-1}) "
              f"aus camera_calibration.json")
        return scale_fn
    elif fit_a != 0.0:
        def scale_fn(v: float) -> float:
            return fit_a * v + fit_b
        print(f"[Infer] Pixelmaßstab: linear (fit_a={fit_a:.2e}, fit_b={fit_b:.6f})")
        return scale_fn
    else:
        def scale_fn(v: float) -> float:
            return fit_b
        print(f"[Infer] Pixelmaßstab: fest {fit_b:.6f} mm/px")
        return scale_fn


# ---------------------------------------------------------------------------
# Sub-Pixel-Zentroid-Verfeinerung (identisch zur Annotationslogik)
# ---------------------------------------------------------------------------

def refine_y_centroid(img_gray: np.ndarray, y_pred_px: float,
                      half_window: int = 30) -> float:
    """
    Sub-Pixel Y-Zentrum per intensitätsgewichtetem Schwerpunkt.

    Findet immer die MITTE der Markierungsbande, nicht deren Kante.
    Identisch zur refine_y_center()-Logik in 2_annotate.py.

    Parameters
    ----------
    img_gray    : Kalibriertes Graustufenbild (H×W, uint8)
    y_pred_px   : Grobe CNN-Vorhersage in Pixeln
    half_window : Suchfenster ±Pixel um die Vorhersage

    Returns
    -------
    float : Verfeinerte Y-Position (sub-pixel)
    """
    h = img_gray.shape[0]
    y0 = max(0, int(y_pred_px) - half_window)
    y1 = min(h, int(y_pred_px) + half_window + 1)
    if y1 <= y0:
        return y_pred_px
    roi        = img_gray[y0:y1, :].astype(np.float32)
    row_means  = roi.mean(axis=1)
    bkg_mean   = row_means.mean()
    weights    = np.abs(row_means - bkg_mean)
    w_sum      = float(weights.sum())
    if w_sum < 1e-9:
        return y_pred_px
    y_local = float(np.dot(weights, np.arange(len(weights))) / w_sum)
    return y0 + y_local


# ---------------------------------------------------------------------------
# Batch-Inferenz
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image_paths: list[str],
    target_size,
    device: torch.device,
    batch_size: int = 32,
    calib_params: dict | None = None,
    rectify_params=None,
    refine_window_px: int = 30,
) -> list[dict]:
    """
    Führt Batch-Inferenz auf allen Bildern durch.

    Wendet optionale Scherungskorrektur (calib_params oder rectify_params) an.
    Verfeinert CNN-Positionen per Sub-Pixel-Zentroid (identisch Annotationslogik).

    Returns
    -------
    Liste von Dicts: {path, y_top_norm, y_bot_norm, y_top_px, y_bot_px,
                      conf_top, conf_bot}
    """
    results = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors     = []
        raw_imgs    = []   # Rohbilder für Zentroid-Verfeinerung
        valid_paths = []

        for p in batch_paths:
            try:
                img = load_and_crop_calibrated(p, calib_params, rectify_params)
            except FileNotFoundError:
                print(f"  Warnung: Bild nicht lesbar: {p}")
                continue
            tensors.append(preprocess(img, target_size))
            raw_imgs.append(img)
            valid_paths.append(p)

        if not tensors:
            continue

        # Sicherstellen dass alle Tensoren gleiche Größe haben
        # (bei target_size=None: alle Bilder in einem Batch müssen gleich groß sein)
        try:
            batch = torch.cat(tensors, dim=0).to(device)
            preds = model(batch)
            pred_list = list(preds)
        except RuntimeError:
            # Größen unterschiedlich → Einzeln verarbeiten
            pred_list = []
            for t in tensors:
                pred_list.append(model(t.to(device))[0])

        for p, img, pred in zip(valid_paths, raw_imgs, pred_list):
            h_img        = img.shape[0]
            y_top_norm   = float(pred[0].item())
            y_bot_norm   = float(pred[1].item())
            conf_top     = float(pred[2].item())
            conf_bot     = float(pred[3].item())

            # CNN-Vorhersage in Pixel umrechnen
            y_top_px_cnn = y_top_norm * h_img
            y_bot_px_cnn = y_bot_norm * h_img

            # Sub-Pixel-Zentroid-Verfeinerung (findet Mitte, nicht Kante)
            y_top_px = refine_y_centroid(img, y_top_px_cnn, refine_window_px)
            y_bot_px = refine_y_centroid(img, y_bot_px_cnn, refine_window_px)

            results.append({
                "path":       p,
                "y_top_norm": y_top_norm,
                "y_bot_norm": y_bot_norm,
                "y_top_px":   y_top_px,    # verfeinert (sub-pixel)
                "y_bot_px":   y_bot_px,    # verfeinert (sub-pixel)
                "conf_top":   conf_top,
                "conf_bot":   conf_bot,
                # Rückwärtskompatibilität: conf = max(conf_top, conf_bot)
                "confidence": max(conf_top, conf_bot),
            })

        pct = min((start + batch_size) / len(image_paths) * 100, 100)
        print(f"\r  {min(start + batch_size, len(image_paths))}/{len(image_paths)} "
              f"({pct:.0f}%)", end="")

    print()
    return results


# ---------------------------------------------------------------------------
# Dehnungsberechnung
# ---------------------------------------------------------------------------

def compute_strain(
    results: list[dict],
    target_h: int | None,
    scale_fn,
    marker_dist_mm: float,
    fps: float,
    confidence_threshold: float = 0.5,
    max_strain: float = 0.15,
) -> list[dict]:
    """
    Berechnet Abstand und Dehnung aus den Inferenzergebnissen.

    Verwendet verfeinerte Pixel-Koordinaten aus run_inference() direkt.
    Wendet Anker+Schätzung-Logik an wenn Abstand physikalisch unplausibel:
      – Toleranz: marker_dist_mm ± max_strain (Standard ±15 %)
      – Konfidenzreicherer Marker = Anker, zweiter wird geschätzt
      – Betroffene Frames werden mit dist_corrected=True markiert

    Parameters
    ----------
    results              : Inferenzergebnisse (aus run_inference)
    target_h             : Bildhöhe in Pixeln (None = aus erstem Bild ermitteln)
    scale_fn             : Funktion v_center_px → mm/px
    marker_dist_mm       : Ruhezustand-Abstand in mm
    fps                  : Framerate
    confidence_threshold : Frames mit conf_bot < threshold werden übersprungen
    max_strain           : Maximale Dehnung [0..1] für Plausibilitätsprüfung
    """
    # Bildhöhe ermitteln falls target_h=None (native Größe)
    if target_h is None and results:
        try:
            sample = load_and_crop_calibrated(results[0]["path"])
            target_h = sample.shape[0]
        except Exception:
            target_h = 1000  # Fallback

    # Nominaler Abstand in Pixeln (für Plausibilitätsprüfung)
    # Verwendet mittlere Skalierung (Bildmitte) als Näherung
    mid_px  = (target_h or 1000) / 2.0
    nom_px  = marker_dist_mm / scale_fn(mid_px)
    dist_min_px = nom_px * (1.0 - max_strain)
    dist_max_px = nom_px * (1.0 + max_strain)

    n_corrected = 0
    output = []

    for i, r in enumerate(results):
        conf_bot = r.get("conf_bot", r.get("confidence", 1.0))
        conf_top = r.get("conf_top", conf_bot)
        base = {
            "frame":      i,
            "filename":   os.path.basename(r["path"]),
            "time_s":     i / fps,
            "conf_top":   float(conf_top),
            "conf_bot":   float(conf_bot),
            # Rückwärtskompatibilität
            "confidence": float(max(conf_top, conf_bot)),
        }

        if conf_bot < confidence_threshold:
            output.append({**base, "y_top_px": None, "y_bot_px": None,
                           "distance_mm": None, "strain_percent": None,
                           "dist_corrected": False})
            continue

        # Pixel-Koordinaten: bevorzuge verfeinerte Werte aus run_inference()
        if "y_bot_px" in r and r["y_bot_px"] is not None:
            y_bot_px = float(r["y_bot_px"])
            y_top_px = float(r["y_top_px"]) if r.get("y_top_px") is not None else None
        else:
            # Fallback: normierte Koordinaten → Pixel
            y_bot_px = r["y_bot_norm"] * target_h
            y_top_px = r["y_top_norm"] * target_h if conf_top > 0.5 else None

        dist_corrected = False

        if y_top_px is not None:
            dist_px = abs(y_bot_px - y_top_px)

            if not (dist_min_px <= dist_px <= dist_max_px):
                # Abstand unplausibel → Anker + physikalische Schätzung
                if conf_bot >= conf_top:
                    # y_bot ist zuverlässiger → y_top schätzen
                    y_top_px = y_bot_px - nom_px
                else:
                    # y_top ist zuverlässiger → y_bot schätzen
                    y_bot_px = y_top_px + nom_px
                dist_corrected = True
                n_corrected += 1
        else:
            # Nur unterer Marker → y_top schätzen
            y_top_px = y_bot_px - nom_px
            dist_corrected = True

        # Sicherstellen dass y_top wirklich über y_bot liegt
        if y_top_px > y_bot_px:
            y_top_px, y_bot_px = y_bot_px, y_top_px

        y_center   = (y_top_px + y_bot_px) / 2.0
        px_size_mm = scale_fn(y_center)
        dist_px    = abs(y_bot_px - y_top_px)
        dist_mm    = dist_px * px_size_mm
        strain_pct = (dist_mm - marker_dist_mm) / marker_dist_mm * 100.0

        output.append({
            **base,
            "y_top_px":       float(y_top_px),
            "y_bot_px":       float(y_bot_px),
            "distance_mm":    float(dist_mm),
            "strain_percent": float(strain_pct),
            "dist_corrected": dist_corrected,
        })

    if n_corrected > 0:
        print(f"[Strain] {n_corrected} Frames: Abstand außerhalb "
              f"[{dist_min_px:.0f}–{dist_max_px:.0f}] px → Anker+Schätzung angewendet")
    return output


# ---------------------------------------------------------------------------
# Physikalische Plausibilitätsfilter
# ---------------------------------------------------------------------------

def apply_physical_filters(
    strain_results: list[dict],
    dist_min_px: float = 50.0,
    dist_max_px: float = 200.0,
    max_jump_px: float = 5.0,
) -> list[dict]:
    """
    Filtert physikalisch unplausible Erkennungen:

    1. Abstandsfilter: |y_bot - y_top| muss in [dist_min_px, dist_max_px] liegen.
    2. Sprungfilter: Positions-Änderung pro Frame ≤ max_jump_px.

    Standard-Werte für 2026-02-26:
        dist_min_px=50, dist_max_px=200 (20 mm Marker, breiter Bereich)
        max_jump_px=5  (konservativ)
    """
    n_dist = n_jump = 0

    # Schritt 1: Abstandsfilter
    for r in strain_results:
        if r["y_top_px"] is None or r["y_bot_px"] is None:
            continue
        dist_px = abs(r["y_bot_px"] - r["y_top_px"])
        if not (dist_min_px <= dist_px <= dist_max_px):
            r["y_top_px"] = None; r["distance_mm"] = None
            r["strain_percent"] = None; r["filter_reason"] = "dist_out_of_range"
            n_dist += 1

    # Schritt 2: Sprungfilter
    prev_top = prev_bot = None
    for r in strain_results:
        if r["y_bot_px"] is not None:
            if prev_bot is not None and abs(r["y_bot_px"] - prev_bot) > max_jump_px:
                r["y_top_px"] = None; r["y_bot_px"] = None
                r["distance_mm"] = None; r["strain_percent"] = None
                r["filter_reason"] = "jump_too_large"
                prev_bot = prev_top = None; n_jump += 1; continue
            prev_bot = r["y_bot_px"]
        else:
            prev_bot = None

        if r["y_top_px"] is not None:
            if prev_top is not None and abs(r["y_top_px"] - prev_top) > max_jump_px:
                r["y_top_px"] = None; r["distance_mm"] = None
                r["strain_percent"] = None
                if "filter_reason" not in r:
                    r["filter_reason"] = "top_jump_too_large"
                prev_top = None; n_jump += 1; continue
            prev_top = r["y_top_px"]
        else:
            prev_top = None

    n_filtered = n_dist + n_jump
    if n_filtered > 0:
        print(f"[Filter] {n_filtered} Frames gefiltert: "
              f"{n_dist} Abstand außer [{dist_min_px:.0f}–{dist_max_px:.0f}] px, "
              f"{n_jump} Sprung > {max_jump_px:.0f} px")
    else:
        print("[Filter] Kein Frame gefiltert.")

    return strain_results


# ---------------------------------------------------------------------------
# 3-Panel-Debug-Video
# ---------------------------------------------------------------------------

def create_debug_video(
    image_paths: list[str],
    strain_results: list[dict],
    output_path: str,
    video_fps: float = 20.0,
    video_step: int = 4,
    calib_params: dict | None = None,
    rectify_params=None,
):
    """
    Erstellt 3-Panel-Debug-Video:
      Panel 1 (links): Aktuelles Bild mit Markierungslinien
      Panel 2 (Mitte): Y-Positionen beider Marker über Zeit
      Panel 3 (rechts): Dehnung in % über Zeit
    """
    if not image_paths or not strain_results:
        print("  Kein Video: keine Daten."); return

    valid = [r for r in strain_results if r["y_top_px"] is not None]
    if not valid:
        print("  Kein Video: keine validen Frames."); return

    all_y = [r["y_top_px"] for r in valid] + [r["y_bot_px"] for r in valid]
    y_min = float(np.percentile(all_y, 2))
    y_max = float(np.percentile(all_y, 98))
    y_pad = max(10.0, (y_max - y_min) * 0.12)
    y_ax_min = max(0.0, y_min - y_pad)
    y_ax_max = y_max + y_pad

    strains = [r["strain_percent"] for r in valid]
    s_min = float(np.percentile(strains, 2))
    s_max = float(np.percentile(strains, 98))
    s_pad = max(0.5, (s_max - s_min) * 0.15)
    s_ax_min, s_ax_max = s_min - s_pad, s_max + s_pad

    x_start = valid[0]["frame"]
    x_max   = strain_results[-1]["frame"]

    try:
        sample = load_and_crop_calibrated(image_paths[0], calib_params, rectify_params)
    except Exception as e:
        print(f"  Video-Fehler: {e}"); return
    h_roi, w_roi = sample.shape[:2]

    scale_p1 = max(1.0, 300.0 / w_roi)
    p1_h = int(h_roi * scale_p1); p1_w = int(w_roi * scale_p1)
    DPI = 80
    FIG_H_PX = max(p1_h, 300)
    p2_w = FIG_H_PX * 2; p3_w = FIG_H_PX * 2
    FIG_W_PX = p1_w + p2_w + p3_w + 60
    fig_w_in = FIG_W_PX / DPI; fig_h_in = FIG_H_PX / DPI
    total_w = p1_w + p2_w + p3_w
    w_ratios = [p1_w / total_w, p2_w / total_w, p3_w / total_w]

    # Test-Frame für Video-Größe
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    out_h, out_w = buf.shape[:2]
    plt.close(fig)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"),
                             video_fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"  Video-Fehler: {output_path}"); return

    indices = list(range(0, len(strain_results), video_step))
    print(f"[Video] {len(indices)} Frames (step={video_step}, fps={video_fps})")

    for vi, idx in enumerate(indices):
        res      = strain_results[idx]
        img_path = image_paths[idx] if idx < len(image_paths) else None

        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI, facecolor="white")
        gs  = fig.add_gridspec(1, 3, width_ratios=w_ratios, wspace=0.45,
                               left=0.03, right=0.98, top=0.88, bottom=0.15)
        ax1, ax2, ax3 = [fig.add_subplot(gs[k]) for k in range(3)]

        # Panel 1: Bild
        has_bot = res["y_bot_px"] is not None
        has_top = res["y_top_px"] is not None
        try:
            img = load_and_crop_calibrated(img_path, calib_params, rectify_params)
            ax1.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=255,
                       interpolation="nearest")
            if has_bot:
                ax1.axhline(y=res["y_bot_px"], color="#FF6600", linewidth=1.5)
            if has_top:
                ax1.axhline(y=res["y_top_px"], color="#00AA00", linewidth=1.5)
        except Exception:
            ax1.set_xlim(0, w_roi); ax1.set_ylim(h_roi, 0)
        conf_str = f"  conf={res['confidence']:.2f}"
        title1 = (f"Frame {res['frame']}  d={res['distance_mm']:.2f}mm{conf_str}" if has_top and has_bot
                  else f"Frame {res['frame']}  [nur unten{conf_str}]" if has_bot
                  else f"Frame {res['frame']}  [kein Marker{conf_str}]")
        ax1.set_title(title1, fontsize=8, pad=3)
        ax1.set_xlabel("Spalte (px)", fontsize=7); ax1.set_ylabel("Zeile (px)", fontsize=7)

        # Panel 2: Y-Positionen
        fr_bot = [r["frame"] for r in strain_results[:idx+1] if r["y_bot_px"] is not None]
        y_bot_ = [r["y_bot_px"] for r in strain_results[:idx+1] if r["y_bot_px"] is not None]
        fr_top = [r["frame"] for r in strain_results[:idx+1] if r["y_top_px"] is not None]
        y_top_ = [r["y_top_px"] for r in strain_results[:idx+1] if r["y_top_px"] is not None]
        if fr_bot: ax2.plot(fr_bot, y_bot_, color="#CC0055", linewidth=0.8, label="y_bot")
        if fr_top: ax2.plot(fr_top, y_top_, color="#0055CC", linewidth=0.8, label="y_top")
        ax2.set_xlim(x_start, x_max); ax2.set_ylim(y_ax_max, y_ax_min)
        ax2.set_title("Markierungs-Position", fontsize=8, pad=3)
        ax2.set_xlabel("Frame", fontsize=7); ax2.set_ylabel("Y-Koordinate (px)", fontsize=7)
        ax2.legend(fontsize=6); ax2.grid(True, alpha=0.3)

        # Panel 3: Dehnung
        fr_s = [r["frame"] for r in strain_results[:idx+1] if r["strain_percent"] is not None]
        s_   = [r["strain_percent"] for r in strain_results[:idx+1] if r["strain_percent"] is not None]
        if fr_s: ax3.plot(fr_s, s_, color="#CC2200", linewidth=0.8)
        ax3.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax3.set_xlim(x_start, x_max); ax3.set_ylim(s_ax_min, s_ax_max)
        s_now = res["strain_percent"]
        ax3.set_title(f"Dehnung: {f'{s_now:+.2f}%' if s_now is not None else '—'}",
                      fontsize=8, pad=3)
        ax3.set_xlabel("Frame", fontsize=7); ax3.set_ylabel("Dehnung (%)", fontsize=7)
        ax3.grid(True, alpha=0.3)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
        plt.close(fig)

        if (vi + 1) % 100 == 0 or vi == len(indices) - 1:
            print(f"\r  Video-Frame {vi+1}/{len(indices)} "
                  f"({(vi+1)/len(indices)*100:.0f}%)", end="")

    writer.release()
    print(f"\nDebug-Video: {output_path}  ({len(indices)/video_fps:.0f} s)")


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    ) if args.device == "auto" else torch.device(args.device)
    print(f"[Infer] Gerät: {device}")

    model, target_size, _ = load_model(args.model, device)
    target_h = target_size[0] if target_size is not None else None

    # Kalibrierung / Rektifizierung laden
    calib_params   = None
    rectify_params = None
    script_dir = Path(__file__).parent

    if args.calib:
        from image_utils import load_camera_calibration
        calib_params = load_camera_calibration(args.calib)
    elif args.rectify:
        from rectify import load_rectification_params
        if args.rectify == "auto":
            cand = script_dir / "rectification_params.json"
            if cand.exists():
                rectify_params = load_rectification_params(cand)
        else:
            rectify_params = load_rectification_params(args.rectify)

    # Pixelmaßstab-Funktion
    scale_fn = get_pixel_scale_fn(calib_params, args.pixel_fit_a, args.pixel_fit_b)

    # Bilder sammeln
    img_folder  = Path(args.images)
    image_paths = sorted([
        str(p) for p in img_folder.iterdir()
        if p.suffix.lower() in (".tif", ".tiff")
    ])
    if not image_paths:
        print(f"Keine .tif Bilder in {img_folder}.", file=sys.stderr); sys.exit(1)
    print(f"[Infer] {len(image_paths)} Bilder gefunden.")

    # Inferenz
    print("[Infer] Starte Inferenz…")
    raw_results = run_inference(model, image_paths, target_size, device, args.batch,
                                calib_params=calib_params, rectify_params=rectify_params)

    # Dehnungsberechnung
    strain_results = compute_strain(
        raw_results, target_h=target_h, scale_fn=scale_fn,
        marker_dist_mm=args.marker_dist_mm, fps=args.fps,
        confidence_threshold=args.confidence_threshold,
        max_strain=args.max_strain,
    )

    # Physikalische Filter
    strain_results = apply_physical_filters(
        strain_results, dist_min_px=args.dist_min_px,
        dist_max_px=args.dist_max_px, max_jump_px=args.max_jump_px,
    )

    # Speichern
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(strain_results, f, indent=2)
    print(f"[Infer] Ergebnisse: {out_path} ({len(strain_results)} Frames)")

    valid = [r for r in strain_results if r["strain_percent"] is not None]
    n_v = len(valid); n_t = len(strain_results)
    print(f"[Infer] Valide Frames: {n_v}/{n_t} ({100*n_v/max(n_t,1):.1f}%)")
    if valid:
        s = [r["strain_percent"] for r in valid]
        print(f"[Infer] Dehnung: min={min(s):+.2f}%  max={max(s):+.2f}%  "
              f"∅={sum(s)/len(s):+.2f}%")

    if args.visualize:
        create_debug_video(
            image_paths, strain_results, str(out_path.with_suffix(".avi")),
            video_fps=args.video_fps, video_step=args.video_step,
            calib_params=calib_params, rectify_params=rectify_params,
        )

    return strain_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferenz: Kettfaden-Markierungs-Tracking (2026-02-26)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",   required=True, help="Modell-Checkpoint (.pt)")
    parser.add_argument("--images",  required=True, help="Ordner mit .tif Bildern")
    parser.add_argument("--output",  default="results.json")
    parser.add_argument("--calib",   type=str, default=None,
                        help="camera_calibration.json (Scherungskorrektur + Maßstab)")
    parser.add_argument("--rectify", type=str, default=None,
                        help="rectification_params.json (älterer Workflow; 'auto' = auto-suchen)")
    parser.add_argument("--fps",    type=float, default=2000.0,
                        help="Kamera-Framerate (Versuch 1-6: 2000; Versuch 8-11: 6000)")
    parser.add_argument("--marker_dist_mm", type=float, default=20.0,
                        help="Markierungsabstand im Ruhezustand (mm)")
    parser.add_argument("--pixel_fit_a", type=float, default=0.0)
    parser.add_argument("--pixel_fit_b", type=float, default=0.2,
                        help="Fester Pixelmaßstab (mm/px) wenn weder --calib noch fit_a")
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--max_strain", type=float, default=0.15,
                        help="Max. Dehnung [0..1] für Anker+Schätzung (Standard: 0.15 = ±15%%)")
    parser.add_argument("--dist_min_px", type=float, default=82.0,
                        help="Minimaler Marker-Abstand (px). 20 mm/0.2065 mm/px×0.85 ≈ 82 px.")
    parser.add_argument("--dist_max_px", type=float, default=112.0,
                        help="Maximaler Marker-Abstand (px). 20 mm/0.2065 mm/px×1.15 ≈ 112 px.")
    parser.add_argument("--max_jump_px", type=float, default=5.0)
    parser.add_argument("--visualize",  action="store_true",
                        help="3-Panel-Debug-Video erzeugen")
    parser.add_argument("--video_step", type=int, default=4)
    parser.add_argument("--video_fps",  type=float, default=20.0)
    args = parser.parse_args()
    main(args)
