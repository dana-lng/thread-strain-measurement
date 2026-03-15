@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  Schritt 2: Annotation der Markierungspositionen (Box-Modus)
echo ============================================================
echo.
echo Belichtungsgruppen:
echo   A: Versuch_1, Versuch_2          (1/6060 s, 2000 fps)
echo   B: Versuch_3-6                   (1/8000 s, 2000 fps)
echo   C: Versuch_8-11                  (1/12500 s, 6000 fps)
echo.
echo Aktueller Versuch unten einstellen: SET VERSUCH=Versuch_X
echo --resume ermoeglicht Unterbrechung und Fortsetzung jederzeit.
echo.

REM ============================================================
REM  HIER ANPASSEN: Gewuenschten Versuch eintragen
REM ============================================================
SET VERSUCH=Versuch_4
SET STEP=20

REM ------------------------------------------------------------
REM  Automatisch abgeleitet (nicht aendern)
REM ------------------------------------------------------------
SET FOLDER=..\%VERSUCH%
SET OUTPUT=..\%VERSUCH%\annotations.json

echo Annotiere : %FOLDER%
echo Ausgabe   : %OUTPUT%
echo Schritt   : jedes %STEP%. Bild
echo.

..\..\nn_marker_tracking\venv\Scripts\python.exe 2_annotate.py ^
    --folder  "%FOLDER%" ^
    --output  "%OUTPUT%" ^
    --calib   camera_calibration.json ^
    --step    %STEP% ^
    --resume ^
    --mode    box

echo.
if %ERRORLEVEL% EQU 0 (
    echo ERFOLG: Annotationen gespeichert in %OUTPUT%
) else (
    echo FEHLER: Annotation fehlgeschlagen.
)
pause
