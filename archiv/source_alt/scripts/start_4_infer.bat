@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  Schritt 4: Inferenz - Gruppe und Versuch auswaehlen
echo ============================================================
echo.
echo  Gruppe A: checkpoints\Group_A\best_model.pt  (Versuch 1+2,  2000 fps)
echo  Gruppe B: checkpoints\Group_B\best_model.pt  (Versuch 3-6,  2000 fps)
echo  Gruppe C: checkpoints\Group_C\best_model.pt  (Versuch 8-11, 6000 fps)
echo.

REM ============================================================
REM  HIER ANPASSEN: Gruppe und Versuchsordner eintragen
REM ============================================================
SET GRUPPE=A
SET VERSUCH=Versuch_1

REM ------------------------------------------------------------
REM  Automatisch abgeleitet (nicht aendern)
REM ------------------------------------------------------------
SET MODEL=checkpoints\Group_%GRUPPE%\best_model.pt
SET FOLDER=..\%VERSUCH%
SET OUTPUT=..\%VERSUCH%\results.json

REM FPS automatisch nach Gruppe setzen
IF "%GRUPPE%"=="C" (SET FPS=6000) ELSE (SET FPS=2000)

echo Gruppe:    %GRUPPE%
echo Modell:    %MODEL%
echo Bilder:    %FOLDER%
echo Ausgabe:   %OUTPUT%
echo Framerate: %FPS% fps
echo.

..\..\nn_marker_tracking\venv\Scripts\python.exe 4_infer.py ^
    --model             "%MODEL%" ^
    --images            "%FOLDER%" ^
    --output            "%OUTPUT%" ^
    --calib             camera_calibration.json ^
    --fps               %FPS% ^
    --marker_dist_mm    20.0 ^
    --max_strain        0.15 ^
    --dist_min_px       82 ^
    --dist_max_px       112 ^
    --max_jump_px       5 ^
    --batch             32 ^
    --visualize

echo.
if %ERRORLEVEL% EQU 0 (
    echo ERFOLG: Ergebnisse in %OUTPUT%
    echo         Debug-Video: %OUTPUT:.json=.avi%
) else (
    echo FEHLER: Inferenz fehlgeschlagen.
    echo Tipp: Modell-Pfad pruefen - muss Training abgeschlossen sein.
)
pause
