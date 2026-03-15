"""
Annotation-Tool für Markierungspositionen auf Kettfaden-Hochgeschwindigkeitsaufnahmen.

Versuchsreihe 2026-02-26: 128×1000 px (Versuch 1-6) und 128×880 px (Versuch 8-11).

Verwendung:
    python 2_annotate.py --folder "Versuch_1" --output Versuch_1/annotations.json
    python 2_annotate.py --folder "Versuch_1" --output Versuch_1/annotations.json \\
                         --calib camera_calibration.json --step 10 --resume

    --folder : Ordner mit .tif Bildern der Versuchsreihe
    --output : Ausgabedatei für Annotationen (JSON)
    --calib  : Pfad zu camera_calibration.json (Scherungskorrektur + Maßstab)
    --step   : Nur jedes N-te Bild annotieren (Standard: 10)
    --resume : Vorhandene Annotationsdatei weiterverwenden (nur fehlende annotieren)
    --mode   : 'box' (Standard) = Rechteck ziehen, 'click' = Y-Linien klicken

Steuerung im Annotationsfenster (--mode box, Standard):
    Drag 1          : Box um UNTERE Markierung ziehen (orange) → Y per Sub-pixel-Schwerpunkt
    Drag 2          : Box um OBERE Markierung ziehen (grün)    → Y per Sub-pixel-Schwerpunkt
    b               : Nur unterer Marker sichtbar (nach Drag 1)
    n               : Kein Marker in diesem Bild (Negativ-Beispiel)
    r               : Aktuelles Bild zurücksetzen (neu annotieren)
    s               : Bild überspringen (keine Annotation gespeichert)
    q               : Beenden und speichern

Steuerung im Annotationsfenster (--mode click):
    Klick 1         : Untere Markierungslinie setzen (orange)
    Klick 2         : Obere Markierungslinie setzen (grün)
    b, n, r, s, q   : wie oben

Ausgabe-Format (JSON):
    {
      "bildname.tif": {
        "y_top": <float>,        Y-Position oberer Marker (Pixel, 0=oben) – 0.0 wenn unsichtbar
        "y_bot": <float>,        Y-Position unterer Marker (Pixel) – sub-pixel Genauigkeit
        "conf_top": <0.0|1.0>,   Konfidenz oberer Marker
        "conf_bot": <0.0|1.0>,   Konfidenz unterer Marker
        "image_width": 128,
        "image_height": 1000
      }, ...
    }

Warum Bounding-Box statt Klick-Linien?
    Das Rechteck-Drag erlaubt eine gezielte Auswahl der Markierungsregion.
    Die genaue Y-Position wird per intensitätsgewichtetem Schwerpunkt (Sub-Pixel-Genauigkeit)
    automatisch berechnet – besser als manuelles Anklicken, da kein menschliches
    Zittern eingeht und die Markierungsmitte genau getroffen wird.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from image_utils import load_and_crop_calibrated


# ---------------------------------------------------------------------------
# Globale Kalibrierungsparameter (werden in main() gesetzt)
# ---------------------------------------------------------------------------

_calib_params  = None   # dict aus camera_calibration.json | None
_rectify_params = None  # RectificationParams | None


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Lädt ein Graustufenbild, croppt es und wendet Scherungskorrektur an."""
    return load_and_crop_calibrated(path, _calib_params, _rectify_params)


def preprocess_for_display(img: np.ndarray) -> np.ndarray:
    """Kontraststreckung für bessere Sichtbarkeit der Markierungen."""
    lo, hi = np.percentile(img, [1, 99])
    img_disp = np.clip(
        (img.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)


def draw_annotations(display: np.ndarray, y_top, y_bot, scale: float) -> np.ndarray:
    """Zeichnet die gesetzten Y-Linien (Klick-Modus)."""
    out = display.copy()
    h, w = out.shape[:2]
    if y_top is not None:
        y = int(round(y_top * scale))
        cv2.line(out, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(out, f"oben y={y_top:.1f}", (5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if y_bot is not None:
        y = int(round(y_bot * scale))
        cv2.line(out, (0, y), (w, y), (0, 100, 255), 1)
        cv2.putText(out, f"unten y={y_bot:.1f}", (5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)
    return out


# ---------------------------------------------------------------------------
# Box-Modus: Sub-pixel Refinement via Bildmomenten
# ---------------------------------------------------------------------------

def refine_y_center(img_gray: np.ndarray,
                    x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Berechnet den sub-pixel Y-Schwerpunkt der Markierung innerhalb der gezogenen Box.

    Strategie: Zeilenweise Mittelwerte bilden, die Abweichung vom Gesamt-
    mittel als Gewicht verwenden. Die Markierung (heller oder dunkler als
    Hintergrund) erzeugt einen Peak bei ihrem Y-Zentrum.

    Returns
    -------
    float : Y-Koordinate in Original-Bildkoordinaten (sub-pixel).
    """
    xi0 = max(0, int(round(min(x0, x1))))
    xi1 = min(img_gray.shape[1], int(round(max(x0, x1))))
    yi0 = max(0, int(round(min(y0, y1))))
    yi1 = min(img_gray.shape[0], int(round(max(y0, y1))))

    if yi1 <= yi0 or xi1 <= xi0:
        return (y0 + y1) / 2.0   # Fallback: Boxmitte

    roi = img_gray[yi0:yi1, xi0:xi1].astype(np.float32)
    row_means = roi.mean(axis=1)
    bkg_mean  = row_means.mean()
    weights   = np.abs(row_means - bkg_mean)
    weights   = np.maximum(weights, 1e-9)

    y_local = float(np.dot(weights, np.arange(len(weights))) / weights.sum())
    return yi0 + y_local


def draw_boxes_display(display: np.ndarray,
                       boxes_orig: list,
                       y_centers: list,
                       scale: float,
                       drag_start_disp,
                       drag_cur_disp) -> np.ndarray:
    """
    Zeichnet fertige Boxen (mit Y-Zentrum-Linie + Kreuz) und aktuelle Drag-Box.

    boxes_orig  : Liste von (x0, y0, x1, y1) in Original-Bildkoordinaten
    y_centers   : Liste von sub-pixel Y-Positionen (je Box eine Zahl)
    drag_*_disp : Aktuelle Drag-Koordinaten in Anzeigekoordinaten (None = kein Drag)
    """
    out = display.copy()
    colors = [(0, 100, 255), (0, 220, 0)]   # orange, grün
    labels = ["unten", "oben"]

    for i, (bx0, by0, bx1, by1) in enumerate(boxes_orig):
        color = colors[i % len(colors)]
        label = labels[i % len(labels)]
        dx0 = int(round(bx0 * scale)); dy0 = int(round(by0 * scale))
        dx1 = int(round(bx1 * scale)); dy1 = int(round(by1 * scale))
        cv2.rectangle(out, (dx0, dy0), (dx1, dy1), color, 1)

        if i < len(y_centers) and y_centers[i] is not None:
            dy_c = int(round(y_centers[i] * scale))
            cv2.line(out, (dx0, dy_c), (dx1, dy_c), color, 2)
            cx = (dx0 + dx1) // 2
            cv2.line(out, (cx - 5, dy_c), (cx + 5, dy_c), (255, 255, 255), 1)
            cv2.line(out, (cx, dy_c - 5), (cx, dy_c + 5), (255, 255, 255), 1)
            cv2.putText(out, f"{label} y={y_centers[i]:.2f}",
                        (dx0 + 3, dy_c - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if drag_start_disp is not None and drag_cur_disp is not None:
        next_i = len(boxes_orig)
        color = colors[next_i % len(colors)]
        sx, sy = drag_start_disp
        cx, cy = drag_cur_disp
        cv2.rectangle(out, (min(sx, cx), min(sy, cy)), (max(sx, cx), max(sy, cy)), color, 1)

    return out


# ---------------------------------------------------------------------------
# Globale Maus-Zustände
# ---------------------------------------------------------------------------

_state = {"clicks": [], "done": False}

_box_state = {
    "start_disp": None,
    "cur_disp":   None,
    "dragging":   False,
    "boxes_orig": [],
}


def _mouse_callback(event, x, y, flags, param):
    """Klick-Modus: Klick setzt Y-Koordinate."""
    scale = param["scale"]
    if event == cv2.EVENT_LBUTTONDOWN:
        _state["clicks"].append((x / scale, y / scale))


def _mouse_callback_box(event, x, y, flags, param):
    """Box-Modus: Drag zeichnet Rechteck."""
    scale = param["scale"]
    if event == cv2.EVENT_LBUTTONDOWN:
        _box_state["start_disp"] = (x, y)
        _box_state["cur_disp"]   = (x, y)
        _box_state["dragging"]   = True
    elif event == cv2.EVENT_MOUSEMOVE and _box_state["dragging"]:
        _box_state["cur_disp"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and _box_state["dragging"]:
        _box_state["dragging"] = False
        sx, sy = _box_state["start_disp"]
        x0_o = min(sx, x) / scale; y0_o = min(sy, y) / scale
        x1_o = max(sx, x) / scale; y1_o = max(sy, y) / scale
        if (x1_o - x0_o) >= 2.0 and (y1_o - y0_o) >= 1.0:
            _box_state["boxes_orig"].append((x0_o, y0_o, x1_o, y1_o))
        _box_state["start_disp"] = None
        _box_state["cur_disp"]   = None


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def annotate(folder: str, output: str, step: int, resume: bool):
    img_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".tif", ".tiff"))
    ])
    if not img_files:
        print(f"Keine .tif Dateien in {folder} gefunden.")
        sys.exit(1)

    # Optional: ab bestimmtem Bild starten
    if args.start is not None:
        start_name = args.start.lower()
        img_files = [f for f in img_files if f.lower() >= start_name]
        print(f"Starte ab '{args.start}': {len(img_files)} Bilder verbleiben.")
        if not img_files:
            print("Keine Bilder gefunden.")
            sys.exit(1)

    # Sampling-Strategie
    step2 = args.step2
    n_total = len(img_files)

    if args.linear:
        if step2 is None:
            step2 = 1
        indices = []
        pos = 0
        while pos < n_total:
            indices.append(pos)
            frac = pos / max(n_total - 1, 1)
            local_step = step * (1 - frac) + step2 * frac
            pos += max(1, int(round(local_step)))
        selected = [img_files[i] for i in indices]
        print(f"{n_total} Bilder, {len(selected)} annotiert (linear {step}→{step2}).")
    elif step2 is None or step2 == step:
        selected = img_files[::step]
        print(f"{n_total} Bilder, {len(selected)} annotiert (jedes {step}.).")
    else:
        mid = n_total // 2
        selected = img_files[:mid:step] + img_files[mid::step2]
        print(f"{n_total} Bilder, {len(selected)} annotiert "
              f"(erste Hälfte jedes {step}., zweite Hälfte jedes {step2}.).")

    # Vorhandene Annotationen laden
    annotations: dict = {}
    if resume and os.path.exists(output):
        with open(output, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        print(f"{len(annotations)} vorhandene Annotationen geladen.")

    # Anzeige-Skalierung
    first_img = load_image(os.path.join(folder, img_files[0]))
    h_orig, w_orig = first_img.shape[:2]
    scale = min(900 / h_orig, 400 / w_orig)
    display_h = int(h_orig * scale)
    display_w = int(w_orig * scale)
    print(f"Originalgröße: {w_orig}×{h_orig} px  |  Anzeige: {display_w}×{display_h} px  "
          f"|  Skalierung: {scale:.2f}×")

    use_box_mode = (args.mode == "box")
    print(f"Annotationsmodus: {'Box-Drag (sub-pixel)' if use_box_mode else 'Klick-Linien'}")

    win = "Annotation – Kettfaden Markierungen"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, display_w, display_h)

    if use_box_mode:
        cv2.setMouseCallback(win, _mouse_callback_box, param={"scale": scale})
        instructions = [
            "Drag 1: Box um UNTERE Markierung (orange)",
            "Drag 2: Box um OBERE Markierung (gruen)  |  b=Nur unten",
            "n=Kein Marker  r=Zuruecksetzen  s=Ueberspringen  q=Beenden",
        ]
    else:
        cv2.setMouseCallback(win, _mouse_callback, param={"scale": scale})
        instructions = [
            "Klick 1: UNTERE Markierungslinie (orange)",
            "Klick 2: OBERE Markierungslinie (gruen)  |  b=Nur unten",
            "n=Kein Marker  r=Zuruecksetzen  s=Ueberspringen  q=Beenden",
        ]

    i = 0
    while i < len(selected):
        fname = selected[i]
        fpath = os.path.join(folder, fname)

        if fname in annotations and resume:
            i += 1
            continue

        try:
            img = load_image(fpath)
        except FileNotFoundError as e:
            print(e)
            i += 1
            continue

        img_gray   = img
        display_base = preprocess_for_display(img)
        display_base = cv2.resize(display_base, (display_w, display_h),
                                  interpolation=cv2.INTER_LINEAR)

        if use_box_mode:
            _box_state.update({"start_disp": None, "cur_disp": None,
                                "dragging": False, "boxes_orig": []})
        else:
            _state["clicks"] = []

        y_top = None; y_bot = None

        print(f"\n[{i+1}/{len(selected)}] {fname}")

        while True:
            if use_box_mode:
                boxes = _box_state["boxes_orig"]
                if len(boxes) >= 1 and y_bot is None:
                    bx0, by0, bx1, by1 = boxes[0]
                    y_bot = refine_y_center(img_gray, bx0, by0, bx1, by1)
                    print(f"  Unten: y_bot={y_bot:.3f} (sub-pixel)")
                if len(boxes) >= 2 and y_top is None:
                    bx0, by0, bx1, by1 = boxes[1]
                    y_top = refine_y_center(img_gray, bx0, by0, bx1, by1)
                    print(f"  Oben:  y_top={y_top:.3f} (sub-pixel)")

                y_centers = ([y_bot] if y_bot is not None else []) + \
                            ([y_top] if y_top is not None else [])
                drag_s = _box_state["start_disp"] if _box_state["dragging"] else None
                drag_c = _box_state["cur_disp"]   if _box_state["dragging"] else None
                disp   = draw_boxes_display(display_base, boxes, y_centers,
                                            scale, drag_s, drag_c)

                if y_bot is None:
                    status = "Drag: Box um UNTERE Markierung  |  n=Kein Marker"
                elif y_top is None:
                    status = "Drag: Box um OBERE Markierung  |  b=Nur unten  n=Kein Marker"
                else:
                    status = "Fertig – naechstes Bild mit beliebiger Taste"
            else:
                if len(_state["clicks"]) >= 1 and y_bot is None:
                    y_bot = _state["clicks"][0][1]
                    print(f"  Unten: y_bot={y_bot:.1f}")
                if len(_state["clicks"]) >= 2 and y_top is None:
                    y_top = _state["clicks"][1][1]
                    print(f"  Oben:  y_top={y_top:.1f}")
                disp   = draw_annotations(display_base, y_top, y_bot, scale)
                status = ("Klick: UNTERE Linie  |  n=Kein Marker" if y_bot is None
                          else "Klick: OBERE Linie  |  b=Nur unten  n=Kein Marker"
                          if y_top is None
                          else "Fertig – naechstes Bild mit beliebiger Taste")

            cv2.putText(disp, status, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            for k, line in enumerate(instructions):
                cv2.putText(disp, line, (5, display_h - 15 - k * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            cv2.putText(disp, f"{fname}  [{i+1}/{len(selected)}]",
                        (5, display_h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1)

            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("n"):
                annotations[fname] = {
                    "y_top": 0.0, "y_bot": 0.0,
                    "image_width": int(w_orig), "image_height": int(h_orig),
                    "conf_top": 0.0, "conf_bot": 0.0,
                }
                print("  Kein Marker gespeichert.")
                i += 1; break

            elif key == ord("b"):
                if y_bot is None:
                    print("  Fehler: Erst untere Markierung definieren.")
                else:
                    annotations[fname] = {
                        "y_top": 0.0, "y_bot": float(y_bot),
                        "image_width": int(w_orig), "image_height": int(h_orig),
                        "conf_top": 0.0, "conf_bot": 1.0,
                    }
                    print(f"  Nur unten: y_bot={y_bot:.3f}")
                    i += 1; break

            elif key == ord("r"):
                if use_box_mode:
                    _box_state.update({"boxes_orig": [], "start_disp": None,
                                       "cur_disp": None, "dragging": False})
                else:
                    _state["clicks"] = []
                y_top = None; y_bot = None
                print("  Zurückgesetzt.")

            elif key == ord("s"):
                print("  Übersprungen.")
                i += 1; break

            elif key == ord("q"):
                print("Beende Annotation.")
                _save(annotations, output)
                cv2.destroyAllWindows()
                return

            elif y_top is not None and y_bot is not None and key != 255:
                if y_top > y_bot:
                    y_top, y_bot = y_bot, y_top
                annotations[fname] = {
                    "y_top": float(y_top), "y_bot": float(y_bot),
                    "image_width": int(w_orig), "image_height": int(h_orig),
                    "conf_top": 1.0, "conf_bot": 1.0,
                }
                print(f"  Gespeichert: y_top={y_top:.3f}, y_bot={y_bot:.3f}")
                i += 1; break

    _save(annotations, output)
    cv2.destroyAllWindows()
    print(f"\nAbgeschlossen. {len(annotations)} Annotationen in '{output}'.")


def _save(annotations: dict, output: str):
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"  Gespeichert: {output} ({len(annotations)} Einträge)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotation-Tool für Kettfaden-Markierungen (2026-02-26)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder",  required=True, help="Ordner mit .tif Bildern")
    parser.add_argument("--output",  default="annotations.json",
                        help="Ausgabe-JSON-Datei")
    parser.add_argument("--calib",   type=str, default=None,
                        help="Pfad zu camera_calibration.json (Y-Scherungskorrektur)")
    parser.add_argument("--rectify", type=str, default=None,
                        help="Alternativ: rectification_params.json (älterer Workflow)")
    parser.add_argument("--step",    type=int, default=10,
                        help="Startabstand: jedes N-te Bild annotieren")
    parser.add_argument("--step2",   type=int, default=None,
                        help="Endabstand (fein). Mit --linear: kontinuierlicher Übergang.")
    parser.add_argument("--linear",  action="store_true",
                        help="Lineares Sampling: Abstand nimmt von --step auf --step2 ab.")
    parser.add_argument("--resume",  action="store_true",
                        help="Vorhandene Annotationen weiterverwenden")
    parser.add_argument("--start",   type=str, default=None,
                        help="Erstes Bild (Dateiname, z.B. '002962.tif')")
    parser.add_argument("--mode",    choices=["box", "click"], default="box",
                        help="Annotationsmodus: 'box' (Drag, Standard) oder 'click'")
    args = parser.parse_args()

    # Kalibrierungsparameter laden (Modul-Scope-Variablen direkt setzen)
    if args.calib:
        from image_utils import load_camera_calibration
        _calib_params = load_camera_calibration(args.calib)
        print(f"[Annotation] Kamerakalibrierung aktiv: {args.calib}")
    elif args.rectify:
        from rectify import load_rectification_params
        _rectify_params = load_rectification_params(args.rectify)
        print(f"[Annotation] Rektifizierung aktiv: {_rectify_params}")

    annotate(args.folder, args.output, args.step, args.resume)
