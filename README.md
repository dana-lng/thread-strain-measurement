# Thread Strain Measurement

Dieses Projekt verarbeitet Hochgeschwindigkeitsbilder aus verschiedenen Versuchen und bietet bisher zwei Hauptschritte:

1. Preprocessing der Rohbilder
   - Scherungskorrektur (`rectify`)
   - Glättung (`smoothing`)
2. Sobel-basierte Markierungserkennung mit Diagrammausgabe
3. Dehnungsberechnung aus Markerabstand und Kalibrierung


## Projektstruktur

```text
thread-strain-measurement
├─ data
│  ├─ calibration
│  ├─ preprocessed
│  └─ raw
├─ figures
│  └─ sobel
├─ src
│  ├─ gui
│  ├─ preprocessesing
│  └─ sobelAlg
├─ requirements.txt
└─ README.md
```


## Virtuelle Umgebung erstellen

Im Projektordner:

```Terminal
python -m venv venv
```

Virtuelle Umgebung aktivieren:

```Terminal
.\venv\Scripts\activate
```


## Abhängigkeiten installieren

Mit aktivierter virtueller Umgebung:

```Terminal
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Datenlogik

- Rohbilder liegen unter `data/raw/Versuch_X`
- Kalibrierung liegt unter `data/calibration`
- Vorverarbeitete Bilder landen unter `data/preprocessed/Versuch_X`
- Sobel-Diagramme landen unter `figures/sobel/Versuch_X`
- Dehnungs-CSV-Dateien landen unter `results/strain/Versuch_X`

Wichtig:
- `preprocessed` enthält bereits das kombinierte Ergebnis aus
  - Scherungskorrektur
  - Glättungmehr


## GUI starten

Die GUI ist der einfachste Einstieg:

```Terminal
python .\src\gui\main_gui.py
```

Die GUI bietet:

- `Preprocess starten`
  - verarbeitet einen kompletten Versuch aus `data/raw/Versuch_X`
  - speichert alle Ergebnisse nach `data/preprocessed/Versuch_X`
- `Sobel starten`
  - läuft auf den bereits preprocesseden Bildern
  - zeigt die Diagramme und die Dehnung live in der GUI
  - speichert die Diagramme nach `figures/sobel/Versuch_X`
  - speichert Messwerte als CSV nach `results/strain/Versuch_X`
- `Abbrechen`
  - stoppt den aktuellen Lauf nach dem gerade bearbeiteten Bild

Eingabe in der GUI:

- nur die Versuchsnummer eingeben, z. B. `1`, `10`, `11`



## Was die Sobel-Ausgabe zeigt

Jedes Diagramm enthält drei Bereiche:

1. preprocessedes Bild
2. Bild mit roten Markierungen
3. Profil-Diagramm der Marker-Suche

Die Marker-Suche arbeitet sequenziell stabilisiert:

- erstes Bild: freie Suche
- folgende Bilder: Suche in einem kleinen Bereich um die zuletzt gefundenen Marker


## Wichtige Dateien

- `src/gui/main_gui.py`
  - Haupt-GUI für Preprocessing und Sobel
- `src/preprocessesing/rectify_first_versuch1.py`
  - kombiniert Scherungskorrektur und Glättung für Einzelbilder
- `src/preprocessesing/rectify_experiment.py`
  - verarbeitet einen kompletten Versuch
- `src/sobelAlg/detect_markers.py`
  - Sobel-Algorithmus und Diagrammerzeugung
- `src/sobelAlg/run_sobel_by_index.py`
  - Sobel-Lauf für das `n`-te Bild eines Versuchs


## Typischer Ablauf

1. `venv` aktivieren
2. `requirements.txt` installieren
3. GUI starten
4. Versuchsnummer eingeben
5. `Preprocess starten`
6. danach `Sobel starten`


## Hinweise

- Die Ordner unter `data` sind in `.gitignore`
- `figures/sobel` darf im Repo existieren, die experimentbezogenen Unterordner darunter bleiben ignoriert
- Für neue Versuche mit anderer Kamerageometrie kann eine neue Kalibrierung nötig sein
