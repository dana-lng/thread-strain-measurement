# Kettfaden-Markierungs-Tracking – Versuchsreihe 2026-02-26

CNN-basierte Erkennung und Dehnungsberechnung von Markierungen auf Kettfäden,
aufgenommen mit einer obliquen Hochgeschwindigkeitskamera.

---

## Kamera-Setup

| Parameter              | Wert                       |
|------------------------|----------------------------|
| Seitlicher Versatz     | 500 mm                     |
| Normaler Abstand       | 330 mm                     |
| Winkel                 | ≈ 56,6°                    |
| Versuch 1–6            | 128 × 1000 px, 2000 fps    |
| Versuch 8–11           | 128 × 880 px,  6000 fps    |
| Markierungsabstand     | 20 mm (Ruhezustand)        |
| Schachbrettfeld        | 10 mm                      |

Durch den Schrägwinkel erscheinen horizontal verlaufende Markierungslinien
im Bild **geneigt**. Die Steigung variiert linear über die Y-Achse:

```
slope(v) = slope_a * v + slope_b
```

Dies wird durch die **Y-Scherungskorrektur** in `rectify.py` kompensiert.
Die Parameter `slope_a` und `slope_b` werden automatisch aus dem
Schachbrettmuster-Bild bestimmt (`1_calibrate_camera.py`).

---

## Schnellstart (4 Schritte)

```
1_calibrate_camera.py  →  camera_calibration.json
2_annotate.py          →  Versuch_X/annotations.json
3_train.py             →  checkpoints/best_model.pt
4_infer.py             →  Versuch_X/results.json  (+  .avi Debug-Video)
```

Alternativ: **BAT-Dateien** doppelklicken (passen Sie zuerst Ordner/FPS an).

### Voraussetzungen

```bash
# Python-Environment installieren (einmalig, aus nn_marker_tracking/)
pip install -r requirements.txt
```

Python-Environment liegt in `../../nn_marker_tracking/venv/`.
Die BAT-Dateien referenzieren es automatisch.

---

## Schritt 1: Kamerakalibrierung

```bash
python 1_calibrate_camera.py \
    --image  "../Massstab/Checkerboard_Pattern_10mm.tif" \
    --square_mm 10 \
    --output camera_calibration.json \
    --verify
```

**Ausgabe:** `camera_calibration.json` mit:
- `shear_model`: `{slope_a, slope_b}` → Y-Scherungskorrektur
- `scale_v_model`: Polynom mm/px als Funktion der Y-Position (vertikal)
- `scale_h_model`: Polynom mm/px als Funktion der Y-Position (horizontal)
- `homography`: 3×3 Matrix für Pixel ↔ Welt-mm Transformation
- `reproj_error_mean_px`: Reprojektionsfehler (sollte < 0,5 px sein)

**Wichtig:** `--inner_cols` und `--inner_rows` entsprechen den **inneren** Ecken
des Schachbretts (Felder minus 1). Beispiel: 3×25-Felder-Brett → 2×24.
Wenn nicht angegeben, wird automatisch erkannt.

**`--verify`** zeigt das Bild mit erkannten Ecken und speichert
`calibration_result.png` mit 4 Analyse-Panels.

### Technischer Hintergrund

Die Homographie H (3×3) bildet Bildpixel (u,v) auf physikalische Koordinaten
(x_mm, y_mm) ab. Über cv2.findHomography (RANSAC) ist sie robust gegen
einzelne fehlerkannte Ecken.

Der Pixelmaßstab variiert über die Y-Achse (Perspektivverzerrung durch
Schrägkamera): Ein Polynom 2. Grades beschreibt mm/px als Funktion von v.

---

## Schritt 2: Annotation

```bash
python 2_annotate.py \
    --folder  "../Versuch_1" \
    --output  "../Versuch_1/annotations.json" \
    --calib   camera_calibration.json \
    --step    10 \
    --resume
```

**Modus `box` (Standard):** Rechteck um die Markierung ziehen →
Sub-pixel Y-Position wird automatisch per **intensitätsgewichtetem Schwerpunkt**
berechnet (genauer als manuelles Anklicken).

**Steuerung:**

| Taste / Aktion | Bedeutung                              |
|----------------|----------------------------------------|
| Drag 1         | Box um **untere** Markierung (orange)  |
| Drag 2         | Box um **obere** Markierung (grün)     |
| `b`            | Nur unterer Marker sichtbar            |
| `n`            | Kein Marker in diesem Bild             |
| `r`            | Aktuelles Bild zurücksetzen            |
| `s`            | Bild überspringen                      |
| `q`            | Beenden und speichern                  |

**Sampling-Optionen:**

```bash
--step 10          # Jedes 10. Bild (gleichmäßig)
--step 20 --step2 5  # Erste Hälfte jedes 20., zweite Hälfte jedes 5.
--step 20 --step2 2 --linear  # Abstand nimmt kontinuierlich ab
--resume           # Fehlende Bilder nachträglich annotieren
```

**Ausgabe-Format (JSON):**
```json
{
  "000001.tif": {
    "y_top": 312.47,   "y_bot": 489.23,
    "conf_top": 1.0,   "conf_bot": 1.0,
    "image_width": 128, "image_height": 1000
  }
}
```

---

## Schritt 3: Training

```bash
# Einzelner Ordner
python 3_train.py \
    --annotations ../Versuch_1/annotations.json \
    --images      ../Versuch_1 \
    --calib       camera_calibration.json

# Mehrere Ordner (empfohlen für besseres Modell)
python 3_train.py \
    --annotations ../Versuch_1/annotations.json ../Versuch_2/annotations.json ... \
    --images      ../Versuch_1 ../Versuch_2 ... \
    --calib       camera_calibration.json \
    --epochs      150 --batch 16
```

**Multi-Folder-Training:** Alle Versuchsordner werden als `ConcatDataset`
kombiniert. Der Train-/Validierungs-Split (85%/15%) erfolgt über den
**gesamten** kombinierten Datensatz.

**Modell-Architektur (MarkerNet):**
- Eingabe: `[B, 1, H, W]` (Graustufenbild, normiert auf [0,1])
- Ausgabe: `[B, 3]` = `[y_top/H, y_bot/H, confidence]`
- `AdaptiveAvgPool2d(1)`: Verarbeitet 128×1000 **und** 128×880 nativ
  (kein Resize nötig, kein Informationsverlust)
- Encoder: 4 Blöcke mit MaxPool + Residual-Verbindungen

**Verlustfunktion (MarkerLoss):**
- L1-Loss für y_top und y_bot (maskengewichtet nach Sichtbarkeit)
- BCE-Loss für confidence
- Ordnungsbedingung: y_top < y_bot (ReLU-Strafe)

**Ausgabe:**
```
checkpoints/
  best_model.pt            ← für Inferenz verwenden
  checkpoint_ep0100.pt     ← Fortsetzung des Trainings
  training_history.json
  learning_curves.png
```

---

## Schritt 4: Inferenz

```bash
python 4_infer.py \
    --model  checkpoints/best_model.pt \
    --images ../Versuch_1 \
    --calib  camera_calibration.json \
    --fps    2000 \
    --output ../Versuch_1/results.json \
    --visualize
```

**Framerate (FPS):**
- Versuch 1–6:  `--fps 2000`
- Versuch 8–11: `--fps 6000`

**Physikalische Filter:**
- `--dist_min_px 50 --dist_max_px 200`: Marker-Abstand außerhalb [50,200] px → verworfen
- `--max_jump_px 5`: Positions-Sprung > 5 px pro Frame → verworfen

**Ausgabe-Format (JSON):**
```json
[
  {
    "frame": 0,
    "filename": "000001.tif",
    "time_s": 0.0005,
    "confidence": 0.97,
    "y_top_px": 312.4,
    "y_bot_px": 489.2,
    "distance_mm": 20.34,
    "strain_percent": 1.70
  }
]
```

Mit `--visualize` wird zusätzlich ein **3-Panel-Debug-Video** erstellt:
- Panel links: Aktuelles Bild mit Markierungslinien
- Panel Mitte: Y-Positionen beider Marker über Zeit
- Panel rechts: Dehnung in % über Zeit

---

## Dateistruktur

```
2026-02-26/
├── code/                          ← Diese Skripte
│   ├── 1_calibrate_camera.py      Kamerakalibrierung (Schachbrett → JSON)
│   ├── 2_annotate.py              Annotation-Tool (Box-Drag, sub-pixel)
│   ├── 3_train.py                 Multi-Folder CNN-Training
│   ├── 4_infer.py                 Batch-Inferenz + Dehnungsberechnung
│   ├── image_utils.py             Bildlade-Hilfsfunktionen
│   ├── dataset.py                 PyTorch Dataset + DataLoader
│   ├── model.py                   MarkerNet CNN-Architektur
│   ├── rectify.py                 Y-Scherungskorrektur-Funktionen
│   ├── requirements.txt           Python-Abhängigkeiten
│   ├── camera_calibration.json    ← wird von Schritt 1 erzeugt
│   ├── checkpoints/               ← wird von Schritt 3 erzeugt
│   ├── start_1_calibrate.bat      BAT-Datei für Schritt 1
│   ├── start_2_annotate.bat       BAT-Datei für Schritt 2
│   ├── start_3_train.bat          BAT-Datei für Schritt 3
│   └── start_4_infer.bat          BAT-Datei für Schritt 4
├── Massstab/
│   └── Checkerboard_Pattern_10mm.tif   Kalibrierungsbild
├── Versuch_1/                     2000 fps, ca. 4700 Bilder
│   ├── *.tif                      128×1000 px Graustufenbilder
│   ├── annotations.json           ← wird von Schritt 2 erzeugt
│   └── results.json               ← wird von Schritt 4 erzeugt
├── Versuch_2/ … Versuch_6/        wie Versuch_1
├── Versuch_8/ … Versuch_11/       6000 fps, 128×880 px
└── …
```

---

## Technische Details für CV-erfahrene Kollegen

### Y-Scherungskorrektur

Durch den 56,6°-Aufnahmewinkel erscheinen horizontale Markierungslinien geneigt.
Die Steigung der Neigung variiert linear mit der Y-Position:

```
slope(v) = slope_a * v + slope_b
```

Korrektur per inversem affinem Warp (cv2.warpAffine):
```
src_y = dst_y + slope(v_center) * (dst_x - u_center)
```

`u_center = 64` (Bildmitte, 128 px / 2): Pixel auf der Mittelachse werden nicht verschoben.

### Sub-Pixel-Annotation (Box-Drag)

Nach dem Ziehen einer Bounding Box wird die genaue Y-Position per
**intensitätsgewichtetem Schwerpunkt** berechnet:

```python
roi = img[yi0:yi1, xi0:xi1]
row_means = roi.mean(axis=1)          # Mittlere Intensität pro Zeile
weights = |row_means - mean(row_means)|  # Abweichung vom Hintergrund
y_subpixel = sum(weights * y_indices) / sum(weights)
```

Dies liefert sub-pixel-genaue Y-Koordinaten unabhängig davon,
ob die Markierung heller oder dunkler als der Hintergrund ist.

### Modell-Training bei gemischten Bildgrößen

Das Modell enthält `AdaptiveAvgPool2d(1)` als letzten Encoder-Layer.
Dadurch werden Features unabhängig von der räumlichen Auflösung auf einen
einzigen globalen Vektor reduziert. Das Modell verarbeitet deshalb
128×1000 (Versuch 1–6) und 128×880 (Versuch 8–11) **ohne Resize**.

Bei Multi-Folder-Training mit gemischten Größen muss jeder Batch
homogene Bildgrößen enthalten (DataLoader macht das automatisch, wenn
jede Versuchsreihe eigenes Dataset hat). Alternativ: `--target_size H W`.

### Pixelmaßstab (positionsabhängig)

Durch die Perspektive variiert der Pixelmaßstab über Y:
```
mm_per_px(v) = polyval(coeffs, v)   # Polynom aus scale_v_model
```

Für die Dehnungsberechnung:
```
dist_mm = |y_bot - y_top| [px] * mm_per_px(y_center)
strain  = (dist_mm - 20.0) / 20.0 * 100  [%]
```

### Homographie (Pixel ↔ Welt-mm)

Die 3×3-Homographiematrix H mappt Bildpixel auf physikalische Koordinaten:
```python
[x_mm, y_mm, 1]^T  ≈  H @ [u_px, v_px, 1]^T
```

Verwendung:
```python
from 1_calibrate_camera import pixel_to_mm, mm_to_pixel
x_mm, y_mm = pixel_to_mm(u, v, calib["homography"])
u, v       = mm_to_pixel(x_mm, y_mm, calib["homography"])
```

---

## Fehlerbehebung

| Problem | Lösung |
|---------|--------|
| Schachbrettecken nicht erkannt | `--verify` nutzen; `--inner_cols` / `--inner_rows` manuell setzen |
| Annotation öffnet sich nicht | OpenCV-GUI-Abhängigkeiten prüfen; venv korrekt aktiviert? |
| Training-Fehler "different sizes" | `--target_size 1000 128` hinzufügen |
| Inferenz: 0% valide Frames | `--dist_min_px` / `--dist_max_px` anpassen; Modell neu trainieren |
| Unicode-Fehler in BAT | `chcp 65001` ist bereits eingebaut |
| CUDA nicht verfügbar | `--device cpu` (langsamer, aber funktioniert) |

---

## Verwandte Skripte

Ältere Versionen (Vorgänger-Messreihen) befinden sich in:
- `../../nn_marker_tracking/` – Messreihe 2021 + 2026-02-13
- `../../HS Kamera/2026-02-23/code/` – Versuchsreihe 2026-02-23 (Schrägkamera, 128×1000)
