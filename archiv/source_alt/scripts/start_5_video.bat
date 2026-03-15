@echo off
chcp 65001 >nul
SET PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ============================================================
echo  Schritt 5: Debug-Video erstellen
echo ============================================================
echo.
echo  Liest results.json und erzeugt 3-Panel-Video:
echo    Links:  Bild mit Marker-Linien (artefaktfrei)
echo    Mitte:  Y-Positionen beider Marker + Glaettungslinie
echo    Rechts: Dehnung in Prozent + Glaettungslinie
echo.

REM ============================================================
REM  HIER ANPASSEN: Gruppe und Versuch eintragen
REM ============================================================
SET GRUPPE=A
SET VERSUCH=Versuch_1

REM  Fenstergroesse des kausalen Rolling-Mean (Frames)
REM  Groesser = glattere Linie, aber mehr Nachlauf
SET SMOOTH=50

REM  Jedes N-te Bild ins Video aufnehmen (4 = Standard, 1 = alle)
SET STEP=4

REM  Video-Framerate (fps)
SET VIDEO_FPS=20

REM ------------------------------------------------------------
REM  Automatisch abgeleitet (nicht aendern)
REM ------------------------------------------------------------
SET RESULTS=..\%VERSUCH%\results.json
SET IMAGES=..\%VERSUCH%
SET OUTPUT=..\%VERSUCH%\debug_video.avi

echo Ergebnisse: %RESULTS%
echo Bilder:     %IMAGES%
echo Ausgabe:    %OUTPUT%
echo Smooth:     %SMOOTH% Frames
echo Step:       %STEP%
echo.

..\..\nn_marker_tracking\venv\Scripts\python.exe create_video.py ^
    --results   "%RESULTS%" ^
    --images    "%IMAGES%"  ^
    --output    "%OUTPUT%"  ^
    --calib     camera_calibration.json ^
    --fps_video %VIDEO_FPS% ^
    --step      %STEP%      ^
    --smooth    %SMOOTH%

echo.
if %ERRORLEVEL% EQU 0 (
    echo ERFOLG: Video gespeichert in %OUTPUT%
    echo Tipp: Mit VLC oder Windows-Mediaplayer oeffnen.
) else (
    echo FEHLER: Video-Erstellung fehlgeschlagen.
    echo Tipp: results.json pruefen - muss Inferenz (Schritt 4) abgeschlossen sein.
)
pause
