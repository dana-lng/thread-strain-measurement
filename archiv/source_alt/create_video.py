#!/usr/bin/env python3
"""
create_video.py – Standalone 3-Panel-Debug-Video aus results.json

Verwendung:
    python create_video.py
        --results  ../Versuch_1/results.json
        --images   ../Versuch_1
        [--output  ../Versuch_1/debug_video.avi]
        [--calib   camera_calibration.json]
        [--fps_video 20]  [--step 4]  [--smooth 50]

Panels:
    Links:  Aktuelles Bild mit Marker-Linien (direkt mit cv2.line gezeichnet)
    Mitte:  Y-Positionen beider Marker über Zeit + geglättete Linie
    Rechts: Dehnung [%] über Zeit + geglättete Linie

Verbesserungen gegenüber der in 4_infer.py integrierten Version:
    - Marker-Linien mit cv2.line() direkt ins Bild-Array gezeichnet
      → keine Artefakte (grüne Pixel/Punkte) durch Matplotlib axhline
    - Kein grüner Trennstrich zwischen Bild- und Graph-Panel
    - Größere Schrift (12–13 pt) + dickere Achsen (1.2 pt)
    - Geglättete Linien (kausaler Rolling-Mean) über Rohdaten überlagert
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Optik-Konstanten ───────────────────────────────────────────────────────────
LABEL_FS   = 12     # Achsenbeschriftung (pt)
TITLE_FS   = 13     # Titel (pt)
TICK_FS    = 10     # Tick-Labels (pt)
LEGEND_FS  = 9      # Legende (pt)
LINE_LW    = 0.8    # Linienbreite Rohdaten
SMOOTH_LW  = 2.5    # Linienbreite geglättet
AXIS_LW    = 1.2    # Achsen-Linienstärke (pt)
RAW_ALPHA  = 0.30   # Transparenz Rohdaten
GRID_A     = 0.30   # Transparenz Gitternetz

DPI = 80            # Video-Rendering-Auflösung

# Farben (Hex für Matplotlib; RGB-Tuple für cv2.line auf RGB-Array)
COLOR_TOP      = "#008800"      # Grün  – oberer Marker
COLOR_BOT      = "#FF6600"      # Orange – unterer Marker
COLOR_STRAIN   = "#CC2200"      # Rotbraun – Dehnung
COLOR_ZERO     = "#888888"      # Grau – Nulllinie
LINE_TOP_RGB   = (0, 136, 0)    # selbe Farbe wie COLOR_TOP als RGB-Tuple
LINE_BOT_RGB   = (255, 102, 0)  # selbe Farbe wie COLOR_BOT als RGB-Tuple
LINE_THICK     = 2              # Liniendicke im Bild (px)


# ── Rolling-Mean (kausal) ─────────────────────────────────────────────────────

def causal_rolling_mean(values: list, window: int) -> list:
    """
    Kausaler Rolling-Mean: jeder Wert verwendet nur vergangene + aktuelle Daten
    (kein Look-ahead → geeignet für Live-Video-Effekt).

    None-Werte werden übersprungen; Ausgabe ist None wenn kein gültiger Wert
    im Fenster vorhanden ist.
    """
    result: list = []
    q: deque = deque()
    for v in values:
        if v is not None:
            q.append(float(v))
            if len(q) > window:
                q.popleft()
            result.append(float(np.mean(q)))
        else:
            result.append(None)
    return result


# ── Bild laden ────────────────────────────────────────────────────────────────

def load_image(path: str, calib_params=None, rectify_params=None) -> np.ndarray:
    """Lädt Bild; nutzt image_utils wenn im selben Verzeichnis vorhanden."""
    try:
        from image_utils import load_and_crop_calibrated
        return load_and_crop_calibrated(path, calib_params, rectify_params)
    except ImportError:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Bild nicht lesbar: {path}")
        return img


# ── Hilfsfunktion: Achsen stylen ──────────────────────────────────────────────

def _style_ax(ax, grid: bool = True):
    """Einheitliches Achsen-Styling: dicke Linien + lesbarer Text."""
    ax.tick_params(labelsize=TICK_FS, width=AXIS_LW, length=4)
    for sp in ax.spines.values():
        sp.set_linewidth(AXIS_LW)
    if grid:
        ax.grid(True, alpha=GRID_A)


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def create_video(
    results: list,
    image_folder: Path,
    output_path: str,
    video_fps: float = 20.0,
    video_step: int  = 4,
    smooth_window: int = 50,
    calib_params=None,
    rectify_params=None,
):
    """
    Erstellt 3-Panel-Debug-Video.

    Parameters
    ----------
    results       : Geladene results.json (Liste von Dicts)
    image_folder  : Ordner mit den .tif Bildern
    output_path   : Ausgabepfad (.avi)
    video_fps     : Video-Framerate
    video_step    : Jedes N-te Ergebnisbild verwenden
    smooth_window : Fenstergröße kausaler Rolling-Mean
    calib_params  : Kalibrierungsdaten (aus image_utils.load_camera_calibration)
    rectify_params: Rektifizierungsparameter (aus rectify.load_rectification_params)
    """
    # ── Valide Frames prüfen ─────────────────────────────────────────────────
    valid = [r for r in results if r.get("y_top_px") is not None]
    if not valid:
        print("Fehler: Keine validen Frames in results.json."); return

    # ── Achsengrenzen (robuste Perzentile, stabil gegen Ausreißer) ───────────
    all_y  = ([r["y_top_px"] for r in valid]
             + [r["y_bot_px"] for r in valid if r.get("y_bot_px") is not None])
    y_min, y_max = float(np.percentile(all_y, 2)), float(np.percentile(all_y, 98))
    y_pad  = max(10.0, (y_max - y_min) * 0.12)
    y_ax_min, y_ax_max = max(0.0, y_min - y_pad), y_max + y_pad

    strains  = [r["strain_percent"] for r in valid]
    s_min, s_max = float(np.percentile(strains, 2)), float(np.percentile(strains, 98))
    s_pad    = max(0.5, (s_max - s_min) * 0.15)
    s_ax_min, s_ax_max = s_min - s_pad, s_max + s_pad

    x_start = results[0]["frame"]
    x_max   = results[-1]["frame"]

    # ── Geglättete Serien vorberechnen (kausal) ──────────────────────────────
    sm_yb = causal_rolling_mean([r.get("y_bot_px")       for r in results], smooth_window)
    sm_yt = causal_rolling_mean([r.get("y_top_px")       for r in results], smooth_window)
    sm_st = causal_rolling_mean([r.get("strain_percent") for r in results], smooth_window)

    # ── Erstes Bild für Dimensionen ──────────────────────────────────────────
    first_name = results[0].get("filename", "")
    first_path = image_folder / first_name if first_name else None
    if first_path is None or not first_path.exists():
        tifs = sorted(image_folder.glob("*.tif"))
        first_path = tifs[0] if tifs else None
    if first_path is None:
        print(f"Fehler: Keine .tif Bilder in {image_folder}"); return

    try:
        sample = load_image(str(first_path), calib_params, rectify_params)
    except Exception as e:
        print(f"Fehler beim Laden des ersten Bildes: {e}"); return

    h_roi, w_roi = sample.shape[:2]

    # ── Panel-Dimensionen ────────────────────────────────────────────────────
    scale_p1 = max(1.0, 300.0 / w_roi)
    p1_h     = int(h_roi * scale_p1)
    p1_w     = int(w_roi * scale_p1)
    FIG_H_PX = max(p1_h, 400)
    p23_w    = int(FIG_H_PX * 1.8)
    FIG_W_PX = p1_w + 2 * p23_w + 80

    fig_w_in = FIG_W_PX / DPI
    fig_h_in = FIG_H_PX / DPI
    total_w  = p1_w + 2 * p23_w
    w_ratios = [p1_w / total_w, p23_w / total_w, p23_w / total_w]

    # Skalierungsfaktor: Bildpixel → Panel-Pixel
    scale_y = p1_h / h_roi

    # ── Video-Writer initialisieren (Dummy-Frame für Größe) ──────────────────
    fig_t = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)
    fig_t.canvas.draw()
    buf_t  = np.asarray(fig_t.canvas.buffer_rgba())[:, :, :3]
    out_h, out_w = buf_t.shape[:2]
    plt.close(fig_t)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        video_fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        print(f"Fehler: Video-Writer konnte nicht geöffnet werden: {output_path}"); return

    indices = list(range(0, len(results), video_step))
    print(f"[Video] {len(indices)} Frames | step={video_step} | "
          f"fps={video_fps:.0f} | smooth_window={smooth_window} | "
          f"Ausgabe: {output_path}")

    # ── Haupt-Loop ────────────────────────────────────────────────────────────
    for vi, idx in enumerate(indices):
        res = results[idx]

        # ── Bild laden + Marker-Linien direkt einzeichnen ───────────────────
        # cv2.line() schreibt direkt ins uint8-Array → keine Matplotlib-Artefakte
        img_disp = None
        img_name = res.get("filename", "")
        img_path = image_folder / img_name if img_name else None

        if img_path is not None and img_path.exists():
            try:
                img_g   = load_image(str(img_path), calib_params, rectify_params)
                img_sc  = cv2.resize(img_g, (p1_w, p1_h),
                                     interpolation=cv2.INTER_LINEAR)
                # Graustufen → RGB (damit cv2.line Farben setzen kann)
                img_rgb = cv2.cvtColor(img_sc, cv2.COLOR_GRAY2RGB)

                # Oberer Marker (grün)
                y_top = res.get("y_top_px")
                if y_top is not None:
                    y_px = int(round(float(y_top) * scale_y))
                    y_px = max(0, min(p1_h - 1, y_px))
                    cv2.line(img_rgb,
                             (0, y_px), (p1_w - 1, y_px),
                             LINE_TOP_RGB, LINE_THICK)

                # Unterer Marker (orange)
                y_bot = res.get("y_bot_px")
                if y_bot is not None:
                    y_px = int(round(float(y_bot) * scale_y))
                    y_px = max(0, min(p1_h - 1, y_px))
                    cv2.line(img_rgb,
                             (0, y_px), (p1_w - 1, y_px),
                             LINE_BOT_RGB, LINE_THICK)

                img_disp = img_rgb
            except Exception:
                pass

        # ── Figure erstellen ─────────────────────────────────────────────────
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI,
                         facecolor="white")
        gs  = fig.add_gridspec(
            1, 3, width_ratios=w_ratios,
            wspace=0.50, left=0.04, right=0.97, top=0.87, bottom=0.14,
        )
        ax1, ax2, ax3 = [fig.add_subplot(gs[k]) for k in range(3)]

        # ── Panel 1: Bild ────────────────────────────────────────────────────
        if img_disp is not None:
            ax1.imshow(img_disp, aspect="auto", interpolation="nearest")
        else:
            ax1.set_xlim(0, p1_w); ax1.set_ylim(p1_h, 0)
            ax1.set_facecolor("#CCCCCC")

        conf  = res.get("confidence", 0.0)
        d_mm  = res.get("distance_mm")
        y_t   = res.get("y_top_px")
        y_b   = res.get("y_bot_px")
        if y_t is not None and y_b is not None:
            t1 = f"Frame {res['frame']}  d={d_mm:.2f} mm  conf={conf:.2f}"
        elif y_b is not None:
            t1 = f"Frame {res['frame']}  [nur unten]  conf={conf:.2f}"
        else:
            t1 = f"Frame {res['frame']}  [kein Marker]  conf={conf:.2f}"

        ax1.set_title(t1, fontsize=TITLE_FS, pad=4)
        ax1.set_xlabel("Spalte (px)", fontsize=LABEL_FS)
        ax1.set_ylabel("Zeile (px)", fontsize=LABEL_FS)
        _style_ax(ax1, grid=False)

        # ── Panel 2: Y-Positionen (kumulativ bis idx) ────────────────────────
        # Rohdaten (dünn, transparent)
        fr_b = [results[i]["frame"] for i in range(idx + 1)
                if results[i].get("y_bot_px") is not None]
        y_b_ = [results[i]["y_bot_px"] for i in range(idx + 1)
                if results[i].get("y_bot_px") is not None]
        fr_t = [results[i]["frame"] for i in range(idx + 1)
                if results[i].get("y_top_px") is not None]
        y_t_ = [results[i]["y_top_px"] for i in range(idx + 1)
                if results[i].get("y_top_px") is not None]

        if fr_b:
            ax2.plot(fr_b, y_b_, color=COLOR_BOT,
                     lw=LINE_LW, alpha=RAW_ALPHA, label="y_bot")
        if fr_t:
            ax2.plot(fr_t, y_t_, color=COLOR_TOP,
                     lw=LINE_LW, alpha=RAW_ALPHA, label="y_top")

        # Geglättete Linien (dick, voll opak)
        smb_x = [results[i]["frame"] for i in range(idx + 1)
                 if sm_yb[i] is not None]
        smb_y = [sm_yb[i] for i in range(idx + 1) if sm_yb[i] is not None]
        smt_x = [results[i]["frame"] for i in range(idx + 1)
                 if sm_yt[i] is not None]
        smt_y = [sm_yt[i] for i in range(idx + 1) if sm_yt[i] is not None]

        if smb_x:
            ax2.plot(smb_x, smb_y, color=COLOR_BOT,
                     lw=SMOOTH_LW, alpha=1.0, label="_nolegend_")
        if smt_x:
            ax2.plot(smt_x, smt_y, color=COLOR_TOP,
                     lw=SMOOTH_LW, alpha=1.0, label="_nolegend_")

        ax2.set_xlim(x_start, x_max)
        ax2.set_ylim(y_ax_max, y_ax_min)   # invertiert: oben = kleine px
        ax2.set_title("Marker-Y-Position", fontsize=TITLE_FS, pad=4)
        ax2.set_xlabel("Frame", fontsize=LABEL_FS)
        ax2.set_ylabel("Y-Koordinate (px)", fontsize=LABEL_FS)
        ax2.legend(fontsize=LEGEND_FS, loc="upper right")
        _style_ax(ax2)

        # ── Panel 3: Dehnung ─────────────────────────────────────────────────
        fr_s = [results[i]["frame"] for i in range(idx + 1)
                if results[i].get("strain_percent") is not None]
        s_   = [results[i]["strain_percent"] for i in range(idx + 1)
                if results[i].get("strain_percent") is not None]
        sms_x = [results[i]["frame"] for i in range(idx + 1)
                 if sm_st[i] is not None]
        sms_y = [sm_st[i] for i in range(idx + 1) if sm_st[i] is not None]

        if fr_s:
            ax3.plot(fr_s, s_, color=COLOR_STRAIN,
                     lw=LINE_LW, alpha=RAW_ALPHA)
        if sms_x:
            ax3.plot(sms_x, sms_y, color=COLOR_STRAIN,
                     lw=SMOOTH_LW, alpha=1.0)

        ax3.axhline(0, color=COLOR_ZERO, lw=1.2, linestyle="--")
        ax3.set_xlim(x_start, x_max)
        ax3.set_ylim(s_ax_min, s_ax_max)

        s_now = res.get("strain_percent")
        ax3.set_title(
            f"Dehnung: {f'{s_now:+.2f}%' if s_now is not None else '—'}",
            fontsize=TITLE_FS, pad=4,
        )
        ax3.set_xlabel("Frame", fontsize=LABEL_FS)
        ax3.set_ylabel("Dehnung (%)", fontsize=LABEL_FS)
        ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        _style_ax(ax3)

        # ── Frame rendern + schreiben ─────────────────────────────────────────
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
        plt.close(fig)

        if (vi + 1) % 50 == 0 or vi == len(indices) - 1:
            pct = (vi + 1) / len(indices) * 100
            print(f"\r  Frame {vi+1}/{len(indices)} ({pct:.0f}%)", end="")

    writer.release()
    n_sec = len(indices) / video_fps
    print(f"\n[Video] Gespeichert: {output_path}  ({n_sec:.1f} s bei {video_fps:.0f} fps)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="3-Panel-Debug-Video aus results.json (standalone)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results",   required=True,
                   help="Pfad zu results.json (Ausgabe von 4_infer.py)")
    p.add_argument("--images",    default=None,
                   help="Ordner mit .tif Bildern "
                        "(Standard: gleicher Ordner wie results.json)")
    p.add_argument("--output",    default=None,
                   help="Ausgabe-Video (.avi; Standard: <results>.avi)")
    p.add_argument("--calib",     default=None,
                   help="camera_calibration.json (Scherungskorrektur)")
    p.add_argument("--rectify",   default=None,
                   help="rectification_params.json (älterer Workflow)")
    p.add_argument("--fps_video", type=float, default=20.0,
                   help="Video-Framerate")
    p.add_argument("--step",      type=int,   default=4,
                   help="Jedes N-te Ergebnisbild ins Video aufnehmen")
    p.add_argument("--smooth",    type=int,   default=50,
                   help="Kausaler Rolling-Mean Fenstergröße (Frames)")
    args = p.parse_args()

    # results.json laden
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Fehler: {results_path} nicht gefunden.", file=sys.stderr)
        sys.exit(1)
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"[Video] {len(results)} Frames aus {results_path.name}")

    # Bildordner bestimmen
    image_folder = Path(args.images) if args.images else results_path.parent
    if not image_folder.is_dir():
        print(f"Fehler: Bilderordner nicht gefunden: {image_folder}",
              file=sys.stderr)
        sys.exit(1)

    # Ausgabepfad
    output_path = args.output or str(results_path.with_suffix(".avi"))

    # Kalibrierung laden (optional)
    calib_params = rectify_params = None
    if args.calib:
        try:
            from image_utils import load_camera_calibration
            calib_params = load_camera_calibration(args.calib)
            print(f"[Video] Kalibrierung geladen: {args.calib}")
        except Exception as e:
            print(f"  Warnung: Kalibrierung nicht geladen: {e}")
    elif args.rectify:
        try:
            from rectify import load_rectification_params
            rectify_params = load_rectification_params(args.rectify)
            print(f"[Video] Rektifizierung geladen: {args.rectify}")
        except Exception as e:
            print(f"  Warnung: Rektifizierung nicht geladen: {e}")

    create_video(
        results, image_folder, output_path,
        video_fps=args.fps_video,
        video_step=args.step,
        smooth_window=args.smooth,
        calib_params=calib_params,
        rectify_params=rectify_params,
    )


if __name__ == "__main__":
    main()
