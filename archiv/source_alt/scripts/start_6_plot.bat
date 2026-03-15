@echo off
chcp 65001 >nul
SET PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ============================================================
echo  Schritt 6: Plots erstellen (PDF + Python-Skript)
echo ============================================================
echo.
echo  Liest results.json und erstellt zwei Graphen:
echo    Y-Position beider Marker ueber Zeit
echo    Dehnung [%%] ueber Zeit
echo.
echo  Jeder Graph wird gespeichert als:
echo    .pdf  fuer direkte Verwendung
echo    .py   eigenstaendiges Skript zum Nachbearbeiten
echo.

REM ============================================================
REM  HIER ANPASSEN: Versuch und Framerate eintragen
REM ============================================================
SET VERSUCH=Versuch_1

REM  Framerate der Kamera (Gruppe A+B: 2000; Gruppe C: 6000)
SET FPS=2000

REM  Fenstergroesse des zentrierten Rolling-Mean (Frames)
REM  Groesser = glattere Linie
SET SMOOTH=100

REM ------------------------------------------------------------
REM  Automatisch abgeleitet (nicht aendern)
REM ------------------------------------------------------------
SET RESULTS=..\%VERSUCH%\results.json
SET PREFIX=..\%VERSUCH%\%VERSUCH%

echo Ergebnisse: %RESULTS%
echo Ausgabe:    %PREFIX%_y_position.pdf  /  %PREFIX%_y_position.py
echo             %PREFIX%_strain.pdf      /  %PREFIX%_strain.py
echo FPS:        %FPS%
echo Smooth:     %SMOOTH% Frames
echo.

..\..\nn_marker_tracking\venv\Scripts\python.exe plot_results.py ^
    --results  "%RESULTS%"  ^
    --fps      %FPS%        ^
    --prefix   "%PREFIX%"   ^
    --smooth   %SMOOTH%

echo.
if %ERRORLEVEL% EQU 0 (
    echo ERFOLG: Plots gespeichert in ..\%VERSUCH%\
    echo.
    echo  Dateien:
    echo    %VERSUCH%_y_position.pdf  ^(Y-Positions-Plot^)
    echo    %VERSUCH%_y_position.py   ^(bearbeitbares Skript^)
    echo    %VERSUCH%_strain.pdf      ^(Dehnungs-Plot^)
    echo    %VERSUCH%_strain.py       ^(bearbeitbares Skript^)
) else (
    echo FEHLER: Plot-Erstellung fehlgeschlagen.
    echo Tipp: results.json pruefen - muss Inferenz (Schritt 4) abgeschlossen sein.
)
pause
