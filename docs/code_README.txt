================================================================================
  KETTFADEN-MARKIERUNGS-TRACKING  –  Technische Dokumentation
  Versuchsreihe 2026-02-26
================================================================================

Eine kompakte README.md mit Schnellstart-Befehlen liegt ebenfalls im selben
Ordner. Diese README.txt erklaert die technischen Hintergruende, die Logik
der Koordinatenumrechnung und alle wichtigen Implementierungsdetails tiefer.

--------------------------------------------------------------------------------
  INHALTSVERZEICHNIS
--------------------------------------------------------------------------------

  1.  Kamera-Setup und geometrische Verhaeltnisse
  2.  Was ist das Ziel des Gesamtsystems?
  3.  Workflow-Uebersicht (4 Schritte)
  4.  Schritt 1: Kamerakalibrierung (1_calibrate_camera.py)
        4a. Was wird kalibriert?
        4b. Was ist ein innerer Schachbretteckpunkt (Sattelpunkt)?
        4c. Eckenerkennung – Methoden und Fallback-Kette
        4d. Ergebnis: camera_calibration.json – alle Felder erklaert
        4e. Aktuelle Kalibrierwerte (Stand: 2026-02-26)
  5.  Koordinatenumrechnung Pixel -> mm (Schritt fuer Schritt)
        5a. Y-Scherungskorrektur (Bildkorrektur)
        5b. Pixelmasstab (positionsabhaengig)
        5c. Dehnungsberechnung
        5d. Homographie (vollstaendige 2D-Umrechnung)
  6.  Schritt 2: Annotation (2_annotate.py)
  7.  Schritt 3: Training (3_train.py / dataset.py / model.py)
  8.  Schritt 4: Inferenz (4_infer.py)
  9.  Abhaengigkeiten zwischen den Dateien
 10.  Haeufige Fehler und Loesungen
 11.  Versionsverlauf der Kalibrierung

--------------------------------------------------------------------------------
  1.  KAMERA-SETUP UND GEOMETRISCHE VERHAELTNISSE
--------------------------------------------------------------------------------

Die Hochgeschwindigkeitskamera ist SEITLICH versetzt aufgestellt:

    Seitlicher Versatz zur Fadenschar:  500 mm
    Normaler Abstand zur Fadenschar:    330 mm
    Aufnahmewinkel (arctan 500/330):    ca. 56.6 deg

Dadurch entstehen ZWEI geometrische Effekte im Bild:

  (A) Y-SCHERUNG (dominant, korrekturbeduertig)
      Real-horizontale Markierungslinien erscheinen geneigt.
      Die Neigung (Steigung dy/dx) variiert LINEARISCH mit der Y-Position
      im Bild:
          slope(v) = slope_a * v + slope_b

      Woher kommt die Y-Abhaengigkeit?
      Die Kamera sieht die Fadenschar unter einem schraegen Winkel. Punkte
      weit oben im Bild (kleine v) entsprechen Fadenpositionen, die raeumlich
      naeher an der Kamera sind – dort ist die perspektivische Verzerrung
      anders als fuer Punkte weit unten (grosse v, weiter weg).

  (B) POSITIONSABHAENGIGER PIXELMASSSTAB
      Ein Pixel entspricht nicht ueberall gleich vielen mm. Durch die
      perspektivische Projektion sind Pixel nahe der Kamera (oben im Bild)
      groesser (mehr mm/px) als Pixel am anderen Ende (unten).
      Beschrieben durch ein Polynom 2. Grades:
          mm_per_px_vertikal(v)   = a*v^2 + b*v + c
          mm_per_px_horizontal(v) = a'*v^2 + b'*v + c'

  (C) LINSENVERZEICHNUNG (vernachlaessigbar)
      Die Kamera hat eine sehr kleine Apertur und grosse Pixel (16 um).
      Radiale Linsenverzeichnung ist bei diesem Kameratyp vernachlaessigbar.
      Es wird KEIN Distortion-Modell verwendet.

Bildformat je nach Versuch:
    Versuch 1-6:   128 x 1000 px,  2000 fps
    Versuch 8-11:  128 x 880  px,  6000 fps

Das Bild ist sehr schmal (128 px) und lang (880/1000 px). Die X-Richtung
zeigt quer zu den Faeden, die Y-Richtung entlang der Faeden.

--------------------------------------------------------------------------------
  2.  WAS IST DAS ZIEL DES GESAMTSYSTEMS?
--------------------------------------------------------------------------------

Ziel: Aus Hochgeschwindigkeitsaufnahmen die lokale DEHNUNG eines Kettfadens
ueber die Zeit messen.

Methode:
  - Auf dem Kettfaden sind zwei Markierungen aufgebracht (Farbpunkte, Tinte
    etc.), im Ruhezustand ca. 20 mm voneinander entfernt.
  - Das CNN (MarkerNet) erkennt pro Frame die Y-Koordinaten beider Markierungen.
  - Der Abstand in Pixeln wird mit dem kalibriertem Massstab in mm umgerechnet.
  - Dehnung = (aktueller Abstand_mm - Ruheabstand_mm) / Ruheabstand_mm * 100 %

Ausgabe: Zeitserie von Dehnungswerten (z.B. alle 0.5 ms bei 2000 fps).

--------------------------------------------------------------------------------
  3.  WORKFLOW-UEBERSICHT (4 SCHRITTE)
--------------------------------------------------------------------------------

  Einmalig (einmalig pro Kamera-Setup):
    1_calibrate_camera.py  --> camera_calibration.json

  Pro Versuchsreihe:
    2_annotate.py          --> Versuch_X/annotations.json   (manuell)
    3_train.py             --> checkpoints/best_model.pt    (Training ~1h)
    4_infer.py             --> Versuch_X/results.json       (automatisch)

BAT-Dateien vorhanden fuer alle 4 Schritte (start_1_calibrate.bat usw.).
Pfad zum Python-Environment: ..\..\nn_marker_tracking\venv\Scripts\python.exe

--------------------------------------------------------------------------------
  4.  SCHRITT 1: KAMERAKALIBRIERUNG (1_calibrate_camera.py)
--------------------------------------------------------------------------------

  4a. WAS WIRD KALIBRIERT?
  -------------------------
  Eingabe: Ein Bild des Schachbrettmusters, aufgenommen mit der Versuchskamera.
           --> ..\Massstab\Checkerboard_Pattern_10mm.tif (128 x 1000 px)
  Feldgroesse des Musters: 10 mm x 10 mm

  Das Skript bestimmt aus dem Schachbrettbild drei Modelle:

    (1) shear_model     --> Y-Scherungskorrektur fuer alle Folgebilder
    (2) scale_v_model   --> Vertikaler Pixelmassstab fuer Dehnungsberechnung
    (3) scale_h_model   --> Horizontaler Pixelmassstab (Dokumentation)
    (4) homography      --> Volle 2D Pixel-zu-mm Transformation (intern)

  Aufruf (aktuell eingestellte Parameter):
    --inner_cols 3   (3 innere Ecken in X-Richtung = 4 Felder breit)
    --inner_rows 20  (20 innere Ecken in Y-Richtung = 21 Felder hoch)
    --square_mm 10
    --verify         (erzeugt calibration_result.png)

  Geometrische Plausibilitaet des Kalibrierungsbildes:
    Breite:  ~4 Felder * 10 mm = 40 mm  bei 128 px --> ca. 0.31 mm/px (horizontal)
    Hoehe:   ~21 Felder * 10 mm = 210 mm bei 1000 px --> ca. 0.21 mm/px (vertikal)
    Diese Werte stimmen mit dem Kalibrierergebnis ueberein (Plausibilitaetscheck).


  4b. WAS IST EIN INNERER SCHACHBRETTECKPUNKT (SATTELPUNKT)?
  ------------------------------------------------------------
  Ein "inner corner" des Schachbretts ist KEIN einfacher Kantenpunkt,
  sondern der SATTELPUNKT, an dem genau 4 Felder zusammentreffen:

        | Schwarz | Weiss |
        |---------|-------|
        | Weiss   | Schwarz |

  An diesem Punkt hat die Intensitaetslandschaft eine SATTELFORM:
  - In eine Richtung (diagonal sw-no): Intensitaetspeak (hell zu dunkel)
  - In die andere Richtung (diagonal nw-so): Intensitaetstal (dunkel zu hell)
  - Gradient = 0 am Sattelpunkt

  OpenCVs cornerSubPix findet diesen Sattelpunkt mit Sub-Pixel-Genauigkeit
  durch analytisches Loesen von:
      Sum_i  (q_i - p) . nabla I(q_i) = 0
  wobei p der gesuchte Sattelpunkt und q_i die Pixel im Suchfenster sind.

  WICHTIG: Der Sattelpunkt liegt NICHT an der Mitte einer Kante, sondern
  am Schnittpunkt zweier diagonaler Feldgrenzen.


  4c. ECKENERKENNUNG - METHODEN UND FALLBACK-KETTE
  --------------------------------------------------
  Das Kalibrierungsbild ist schwierig weil:
    - Nur 128 px breit (sehr schmal)
    - Stark geneigt (56.6 deg Kamerawinkel)
    - Nur 3 innere Spalten (Grenzfall fuer OpenCV)

  Deshalb versucht detect_corners() drei Methoden in dieser Reihenfolge:

  Methode 1: cv2.findChessboardCornersSB()  [BEVORZUGT]
    - "SB" = Saddle-point Based (Sattelpunkt-basiert)
    - Findet direkt die 2x2-Sattelpunkte ohne Intensitaetsschwellwert
    - Robuster bei Scherung, kleinen Mustern und variierendem Kontrast
    - Flags: NORMALIZE_IMAGE | EXHAUSTIVE | ACCURACY
    - ACCURACY integriert bereits Sub-Pixel-Genauigkeit intern
    - Benoetigt mind. 3x3 innere Ecken

  Methode 2: cv2.findChessboardCorners() + cornerSubPix(11,11)
    - Klassische Methode (Intensitaetsbasiert)
    - Suchfenster vergroessert auf (11,11) statt Standard (7,7)
    - Groesseres Fenster noetig weil Scherung die X-Position
      pro Zeile um bis zu ~10 px vom Profilmittelwert abweichen laesst
    - Benoetigt mind. 3x3 innere Ecken

  Methode 3: _corners_via_profile() (Fallback fuer inner_cols < 3)
    - Sobel-Kantenprojektionsbasiert (NICHT Intensitaetsprofil!)
    - Intensitaetsprofil ist fuer Schachbrett FLACH (je Spalte ~50% sw/weiss)
    - Sobel hat klare Peaks an Feldubergaengen
    - Rand-Filter: Peaks < 0.6 * Feldbreite vom Rand ausgeschlossen
      (verhindert Erkennung des Schachbrett-Aussenrands als "innere Ecke")
    - Anschliessend cornerSubPix(11,11) zur Sattelpunkt-Verfeinerung


  4d. ERGEBNIS: camera_calibration.json - ALLE FELDER ERKLAERT
  --------------------------------------------------------------
  {
    "square_mm": 10.0,         Feldgroesse des Kalibrierungsmusters in mm

    "inner_cols": 3,           Erkannte innere Spaltenecken (= Felder-1 in X)
    "inner_rows": 20,          Erkannte innere Reihenecken (= Felder-1 in Y)

    "image_width": 128,        Bildbreite in Pixeln
    "image_height": 1000,      Bildhoehe in Pixeln
    "u_center": 64.0,          Bildmitte in X (Zentrum der Scherungskorrektur)

    "calibration_image": "...", Pfad zum verwendeten Kalibrierungsbild

    "corners_px": [[u,v], ...], Erkannte Eckpositionen in Bildpixeln
                                Reihenfolge: col 0 rechts (groesses u),
                                col (inner_cols-1) links (kleinstes u),
                                Zeilen von unten (groesse v) nach oben

    "world_pts_mm": [[x,y],...], Zugehoerige Weltkoordinaten in mm
                                 x: horizontal 0/10/20 mm
                                 y: vertikal 0/10/.../190 mm

    "homography": [[...],[...],[...]]  3x3 Matrix H:
                                 [x_mm, y_mm, 1]^T ~ H @ [u_px, v_px, 1]^T
                                 Berechnet via RANSAC aus corners_px <-> world_pts_mm

    "reproj_error_mean_px": ..., Mittlerer Reprojektionsfehler (px)
                                 Sollte < 0.5 px sein
                                 WICHTIG: Restfehler > 0 entsteht durch die
                                 Nichtlinearitaet (positionsabh. Scherung),
                                 die eine lineare Homographie nicht exakt abbildet.

    "reproj_error_max_px": ...,  Maximaler Reprojektionsfehler (px)

    "shear_model": {
      "slope_a": ...,    Steigung der Steigungsfunktion [1/px]
      "slope_b": ...,    Steigung bei v=0 (oberer Bildrand) [-]
      "rmse": ...,       Streuung der gemessenen Zeilensteigungen [px/px]
                         MASSGEBEND: sollte < 0.01 sein
      "n_points": 20     Anzahl der Messpunkte (= inner_rows)
    },

    "scale_v_model": {
      "degree": 2,       Polynomgrad
      "coeffs": [a,b,c], Koeffizienten fuer np.polyval(coeffs, v)
                         Einheit: mm/px (an Stelle v in Pixeln)
      "rmse": ...,       Streuung der lokalen Massstabsmessungen [mm/px]
      "n_points": 57,    Anzahl benachbarter Eckpaare in Y
      "mean_mm_per_px":  Mittlerer Massstab ueber alle Y-Positionen
    },

    "scale_h_model": {   Analog fuer horizontale Richtung
      ...
      "n_points": 40     Anzahl benachbarter Eckpaare in X
    }
  }


  4e. AKTUELLE KALIBRIERWERTE (Stand: letzter Kalibrierungslauf 2026-02-26)
  ---------------------------------------------------------------------------
  Verwendete Methode: cv2.findChessboardCornersSB (Sattelpunkt-basiert)
  inner_cols: 3   inner_rows: 20

  Reprojektionsfehler:
    Mittelwert:  0.424 px   (hervorragend, < 0.5 px)
    Maximum:     1.265 px

  Scherungsmodell:
    slope_a =  4.701e-4  [1/px]   (wie stark die Steigung mit Y zunimmt)
    slope_b = -0.2939    [-]      (Steigung bei v=0, oberer Bildrand)
    rmse    =  0.0047 px/px       (sehr gut, < 0.01 angestrebt)
    Interpretation:
      Bei v=0   (Bildrand oben):  slope = -0.294  (starke Neigung nach links-oben)
      Bei v=625 px:               slope =  0.000  (keine Neigung, Wendepunkt)
      Bei v=1000 (Bildrand unten): slope = +0.176  (Neigung nach rechts-unten)
    D.h. Linien im mittleren Bildbereich (~v=625) sind bereits horizontal.

  Vertikaler Pixelmassstab:
    Polynom: mm_per_px(v) = -3.587e-9*v^2 - 8.231e-6*v + 0.2116
    Mittelwert: 0.2065 mm/px
    Plausibilitaet: 1000 px * 0.207 mm/px = 207 mm ~ 21 Felder * 10 mm = 210 mm  OK

  Horizontaler Pixelmassstab:
    Polynom: mm_per_px_h(v) = 3.259e-8*v^2 - 4.282e-5*v + 0.3503
    Mittelwert: 0.340 mm/px
    Plausibilitaet: 128 px * 0.340 mm/px = 43.5 mm ~ 4.5 Felder * 10 mm = 45 mm  OK

  Homographie (H):
    [-0.354  -0.0087   39.41 ]
    [ 0.034  -0.214   202.60 ]
    [ 5.1e-4  4.3e-5    1.0  ]
    Bedeutung H[0][0] = -0.354: steigendes u -> fallendes x_mm (x=0 rechts im Bild)
    Bedeutung H[1][1] = -0.214: steigendes v -> fallendes y_mm (y=0 unten im Bild)

--------------------------------------------------------------------------------
  5.  KOORDINATENUMRECHNUNG PIXEL -> MM (SCHRITT FUER SCHRITT)
--------------------------------------------------------------------------------

  5a. Y-SCHERUNGSKORREKTUR (BILDKORREKTUR)
  -----------------------------------------
  Angewendet in: image_utils.load_and_crop_calibrated()
                 --> rectify.rectify_image()

  Was passiert: Das Eingabebild wird geometrisch transformiert (cv2.warpAffine),
  sodass real-horizontale Linien auch im korrigierten Bild horizontal sind.

  Formel (inverse affine Abbildung, pro Pixel):
    src_y = dst_y + slope(v_center) * (dst_x - u_center)
    src_x = dst_x                                         (keine x-Verschiebung)

  wobei:
    slope(v_center) = slope_a * v_center + slope_b
    v_center  = Y-Koordinate des Ziel-Pixels
    u_center  = 64.0 (Bildmitte, Pixel auf der Mittelachse werden nicht bewegt)
    src_x/y   = Koordinate im Originalbild (von dort wird interpoliert)
    dst_x/y   = Koordinate im korrigierten Ausgabebild

  Alle nachfolgenden Y-Koordinaten (Annotationen, Modell-Output, Ergebnis-JSON)
  beziehen sich auf das KORRIGIERTE Bild.

  Wann wird das gemacht?
    - In MarkerDataset.__getitem__() vor dem Training
    - In 4_infer.py vor der Modell-Inferenz
    - In 2_annotate.py vor der Anzeige zum Annotieren
    --> Konsistenz: Modell sieht korrigierte Bilder, gibt korrigierte Koordinaten aus.


  5b. PIXELMASSSTAB (POSITIONSABHAENGIG)
  ---------------------------------------
  Verwendet in: 4_infer.py --> get_pixel_scale_fn()

  Die detektierte Y-Position eines Markers (z.B. y_bot = 480.5 px) wird wie
  folgt in Millimeter umgerechnet:

    y_center_px = (y_top + y_bot) / 2         Mittelpunkt zwischen den Markern
    mm_per_px   = polyval(coeffs, y_center_px)  Massstab an dieser Y-Position
    dist_mm     = |y_bot - y_top| [px] * mm_per_px

  Das Polynom ist in scale_v_model["coeffs"] gespeichert und wird mit
  numpy.polyval(coeffs, v) ausgewertet.

  Warum positionsabhaengig?
    Durch die Perspektive der Schraegkamera stellt ein Pixel bei v=100
    mehr mm dar als bei v=900. Das Polynom beschreibt diese Abhaengigkeit
    mit quadratischer Genauigkeit.


  5c. DEHNUNGSBERECHNUNG
  -----------------------
  In 4_infer.py:

    dist_mm_aktuell = |y_bot - y_top| [px] * mm_per_px(y_center_px)
    dist_mm_ruhe    = 20.0 mm  (Vorgabe per --marker_dist_mm, Standardwert)
    strain_percent  = (dist_mm_aktuell - dist_mm_ruhe) / dist_mm_ruhe * 100

  Ausgabe im results.json pro Frame:
    "distance_mm":    aktueller Markierungsabstand in mm
    "strain_percent": Dehnung in Prozent (0 im Ruhezustand)


  5d. HOMOGRAPHIE (VOLLSTAENDIGE 2D-UMRECHNUNG)
  -----------------------------------------------
  Verwendet in: 1_calibrate_camera.py intern (Reprojektionsfehler-Berechnung)
  NICHT direkt in Annotation, Training oder Inferenz benutzt.

  Gibt fuer jeden Bildpunkt (u, v) die vollstaendige physikalische
  Position (x_mm, y_mm) im Schachbrett-Koordinatensystem an.

  Aufruf ueber oeffentliche API in 1_calibrate_camera.py:
    from _1_calibrate_camera import pixel_to_mm, mm_to_pixel
    x_mm, y_mm = pixel_to_mm(u_px, v_px, calib["homography"])
    u_px, v_px = mm_to_pixel(x_mm, y_mm, calib["homography"])

  Warum wird die Homographie NICHT fuer die Dehnungsmessung benutzt?
    Die Homographie ist eine lineare Transformation. Sie kann die
    positionsabhaengige Scherung (slope(v) variiert mit v) nicht exakt
    abbilden. Deshalb ist der direkte Weg (Korrektur via shear_model +
    Massstab via scale_v_model) genauer.

  Wann ist die Homographie nuetzlich?
    Wenn man fuer einen einzelnen Bildpunkt sowohl x_mm als auch y_mm
    braucht (z.B. um eine absolute Position im Raum zu bestimmen).

--------------------------------------------------------------------------------
  6.  SCHRITT 2: ANNOTATION (2_annotate.py)
--------------------------------------------------------------------------------

Zweck: Manuelles Markieren der beiden Faden-Markierungen in einer Stichprobe
       von Bildern. Diese Annotationen dienen als Trainingsdaten fuer das CNN.

Wie annotieren:
  - Standard-Modus: Rechteck um die Markierung ziehen (--mode box, Standard)
  - Die exakte Y-Position wird automatisch per intensitaetsgewichtetem
    Schwerpunkt berechnet (Sub-Pixel-Genauigkeit, robuster als manuelles Klicken):
        roi = img[yi0:yi1, xi0:xi1]
        row_means = roi.mean(axis=1)
        weights = abs(row_means - mean(row_means))
        y_subpixel = sum(weights * y_indices) / sum(weights)

Sampling-Strategie empfohlen:
  - --step 10 --resume  fuer normales Training
  - --step 5 oder --step 2 fuer kritische Versuchsphasen (starke Dehnung)
  - Ungefaehr 50-200 annotierte Frames pro Versuchsreihe genuegen als Start

Ausgabe-Format (annotations.json):
  { "000001.tif": { "y_top": 312.47, "y_bot": 489.23,
                    "conf_top": 1.0, "conf_bot": 1.0,
                    "image_width": 128, "image_height": 1000 }, ... }
  conf = 0.0 wenn der Marker in diesem Frame nicht sichtbar ist (Negativ-Beispiel)

--------------------------------------------------------------------------------
  7.  SCHRITT 3: TRAINING (3_train.py / dataset.py / model.py)
--------------------------------------------------------------------------------

  MODELL-ARCHITEKTUR (model.py):
  -------------------------------
  MarkerNet: Eingabe [B,1,H,W] --> Ausgabe [B,3]

  Encoder (4 Bloecke):
    Block 1: ConvBnReLU(1,32) + MaxPool(2,2)
    Block 2: ConvBnReLU(32,64) + MaxPool(2,2)
    Block 3: ConvBnReLU(64,128) + ResBlock(128) + MaxPool(2,2)
    Block 4: ConvBnReLU(128,256) + ResBlock(256)
    --> AdaptiveAvgPool2d(1)  [globales Pooling, unabhaengig von H und W]

  Ausgabe-Kopf:
    Linear(256, 3) --> Sigmoid --> [y_top/H, y_bot/H, confidence]

  Warum AdaptiveAvgPool2d(1)?
    Damit verarbeitet ein einziges trainiertes Modell sowohl 128x1000-Bilder
    (Versuch 1-6) als auch 128x880-Bilder (Versuch 8-11) OHNE Resize.

  VERLUSTFUNKTION (MarkerLoss in model.py):
    L = w_top * L1(y_top_pred, y_top_gt) * conf_top_gt    (Y-Regression)
      + w_bot * L1(y_bot_pred, y_bot_gt) * conf_bot_gt    (Y-Regression)
      + w_conf * BCE(conf_pred, conf_gt)                   (Konfidenz)
      + w_ord * max(0, y_top_pred - y_bot_pred + margin)  (Reihenfolge)

  DATA-AUGMENTIERUNG (dataset.py):
    - Gauss'sches Rauschen (sigma=0.02)
    - Helligkeit +/-15%
    - Horizontale Spiegelung (nur wenn kein physikalischer Unterschied)
    - KEINE vertikale Spiegelung (aendert relative Markerposition)

  MULTI-FOLDER-TRAINING:
    build_combined_dataloaders() kombiniert alle Versuchsordner zu einem
    einzigen ConcatDataset. Train-/Val-Split 85%/15% ueber alle Daten gemeinsam.

  Hyperparameter (Empfehlung):
    --epochs 100 --batch 16 --lr 1e-3
    LR-Schedule: CosineAnnealingLR(T_max=epochs)

  Ausgabe:
    checkpoints/best_model.pt           --> fuer Inferenz
    checkpoints/training_history.json   --> Loss/MAE pro Epoche
    checkpoints/learning_curves.png     --> Plot

--------------------------------------------------------------------------------
  8.  SCHRITT 4: INFERENZ (4_infer.py)
--------------------------------------------------------------------------------

Ablauf pro Frame:
  1. load_and_crop_calibrated(path, calib_params)
     --> Bild laden + crop_to_roi + Y-Scherungskorrektur
  2. MarkerNet(image_tensor) --> [y_top/H, y_bot/H, confidence]
  3. Physikalische Filter (Abstand min/max, Sprung max)
  4. dist_mm = (y_bot - y_top) [px] * mm_per_px(y_center_px)
  5. strain_percent = (dist_mm - 20.0) / 20.0 * 100

Wichtige Parameter:
  --fps 2000        fuer Versuch 1-6  (Zeitstempel: t = frame / fps)
  --fps 6000        fuer Versuch 8-11
  --marker_dist_mm  Ruhe-Markierungsabstand (Standard 20 mm)
  --dist_min_px 50  Minimum Markierungsabstand in Pixeln
  --dist_max_px 200 Maximum Markierungsabstand in Pixeln
  --max_jump_px 5   Max. Positionssprung pro Frame (physikalischer Filter)

Visualisierung (--visualize):
  Erzeugt ein .avi Debug-Video mit 3 Panels:
    Links:  Aktuelles Bild + Markierungslinien
    Mitte:  Y-Positionen beider Marker ueber Zeit
    Rechts: Dehnung in % ueber Zeit

Ausgabe results.json:
  [{ "frame": 0, "filename": "000001.tif", "time_s": 0.0005,
     "confidence": 0.97, "y_top_px": 312.4, "y_bot_px": 489.2,
     "distance_mm": 20.34, "strain_percent": 1.70 }, ...]

--------------------------------------------------------------------------------
  9.  ABHAENGIGKEITEN ZWISCHEN DEN DATEIEN
--------------------------------------------------------------------------------

  camera_calibration.json wird von:
    image_utils.py --> load_camera_calibration()
      --> shear_model: slope_a, slope_b
          Verwendet in load_and_crop_calibrated() -> rectify_image()
          Aufgerufen von: dataset.py, 2_annotate.py, 4_infer.py
      --> scale_v_model: coeffs (Polynom mm/px)
          Verwendet in 4_infer.py -> get_pixel_scale_fn()
      --> homography: [3x3]
          Verwendet intern in 1_calibrate_camera.py (nicht von anderen Skripten)

  Importkette:
    2_annotate.py   --> image_utils (load_and_crop_calibrated)
    3_train.py      --> dataset (build_dataloaders)
                    --> dataset.py --> image_utils (load_and_crop_calibrated)
                    --> model.py (build_model)
    4_infer.py      --> model.py (build_model)
                    --> image_utils (load_and_crop_calibrated)

  Dateien ohne externe Abhaengigkeiten (Basismodule):
    rectify.py      --> nur cv2, numpy
    image_utils.py  --> cv2, numpy, rectify (optional), 1_calibrate_camera (optional)

  Zusammenfassung: Was aus camera_calibration.json wirklich entscheidend ist:
    shear_model         --> Bildkorrektur, konsistent durch den ganzen Workflow
    scale_v_model       --> Pixel-zu-mm fuer Dehnungsmessung
    scale_h_model       --> nur Dokumentation / Plausibilitaetspruefung
    homography          --> nur in Skript 1 intern (Kalibrierungsvalidierung)

--------------------------------------------------------------------------------
  10.  HAEUFIGE FEHLER UND LOESUNGEN
--------------------------------------------------------------------------------

  Problem: Schachbrettecken nicht erkannt / falscher reproj_error
  ---------------------------------------------------------------
  - --verify verwenden um calibration_result.png zu pruefen
  - --inner_cols und --inner_rows manuell angeben (nicht auto)
  - Gezaehlte Felder im Bild: X-Felder - 1 = inner_cols, Y-Felder - 1 = inner_rows
  - findChessboardCornersSB benoetigt min. 3x3 innere Ecken
  - Fuer inner_cols < 3: automatisch Sobel-Profil-Methode (Fallback)

  Problem: reproj_error > 1 px trotz korrekt erkannter Ecken
  ----------------------------------------------------------
  - Ein gewisser Restfehler (0.3-0.5 px) ist NORMAL, da die lineare
    Homographie die positionsabhaengige Scherung nicht exakt abbilden kann.
  - Entscheidender Qualitaetsindikator: shear_model.rmse (sollte < 0.01 sein)
  - scale_v_model.mean_mm_per_px sollte zum geometrischen Plausibilitaetswert passen

  Problem: Training-Fehler "different sizes in batch"
  ---------------------------------------------------
  - Versuch 1-6 (128x1000) und Versuch 8-11 (128x880) koennen nicht im gleichen Batch
  - Loesung A: Separat trainieren (bevorzugt)
  - Loesung B: --target_size 1000 128 (skaliert alle auf gleiche Groesse)

  Problem: Inferenz: sehr wenige oder keine validen Frames
  --------------------------------------------------------
  - --dist_min_px und --dist_max_px anpassen (Standardwerte evtl. zu eng)
  - Modell neu trainieren mit mehr Annotationen
  - --visualize einschalten fuer Debug-Video

  Problem: Unicode-Fehler in BAT-Dateien
  ---------------------------------------
  - chcp 65001 ist bereits in allen BAT-Dateien vorhanden (UTF-8 Konsolausgabe)

  Problem: CUDA nicht verfuegbar
  ------------------------------
  - Das Skript faellt automatisch auf CPU zurueck
  - Explizit: --device cpu (Training deutlich langsamer)

--------------------------------------------------------------------------------
  11.  VERSIONSVERLAUF DER KALIBRIERUNG
--------------------------------------------------------------------------------

  Iteration 1 (auto-detect):
    Ergebnis: inner_cols=6, inner_rows=65, reproj=15.4 px
    Problem: Zu viele Sobel-Peaks durch intensitaetsbasierte Profil-Methode

  Iteration 2 (--inner_cols 4 --inner_rows 20):
    Ergebnis: reproj=6.29 px, scale_h=0.411 mm/px (zu gross)
    Problem: Falscher Peak bei x=17 (Schachbrett-Aussenrand erkannt als innere Ecke)

  Iteration 3 (--inner_cols 3, Rand-Filter 0.5 * Feldbreite):
    Ergebnis: reproj=8.49 px (schlechter)
    Problem: Rand-Filter margin=16 px, x=17 gerade nicht gefiltert

  Iteration 4 (Rand-Filter auf 0.6 * Feldbreite erhoehen):
    Ergebnis: reproj=3.03 px, scale_v=0.209, scale_h=0.310
    Besser, aber cornerSubPix-Anfangsschaetzung oft > 7 px vom Sattelpunkt

  Iteration 5 (findChessboardCornersSB + cornerSubPix-Fenster 11x11):
    Ergebnis: reproj=0.424 px, shear rmse=0.0047
    Methode: findChessboardCornersSB (Sattelpunkt-basiert) erfolgreich
    --> AKTUELLER STAND, Produktionsqualitaet

================================================================================
  Ende der Dokumentation
================================================================================
