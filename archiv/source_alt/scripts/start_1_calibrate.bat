@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  Schritt 1: Kamerakalibrierung aus Schachbrettmuster-Bild
echo ============================================================
echo.
echo Eingabe:  ..\Massstab\Checkerboard_Pattern_10mm.tif
echo Ausgabe:  camera_calibration.json
echo.
echo Optionen:
echo   --square_mm 10     Feldgroesse des Schachbretts in mm
echo   --inner_cols 2     Anzahl innerer Ecken in X (oder weglassen = auto)
echo   --inner_rows 24    Anzahl innerer Ecken in Y (oder weglassen = auto)
echo   --verify           Visualisierung anzeigen und als PNG speichern
echo.

..\..\nn_marker_tracking\venv\Scripts\python.exe 1_calibrate_camera.py ^
    --image      "..\Massstab\Checkerboard_Pattern_10mm.tif" ^
    --square_mm  10 ^
    --inner_cols 3 ^
    --inner_rows 20 ^
    --output     camera_calibration.json ^
    --verify

echo.
if %ERRORLEVEL% EQU 0 (
    echo ERFOLG: camera_calibration.json gespeichert.
) else (
    echo FEHLER: Kalibrierung fehlgeschlagen.
    echo Moegliche Ursache: Schachbrettecken nicht erkannt.
    echo Tipp: --inner_cols und --inner_rows manuell angeben.
)
pause
