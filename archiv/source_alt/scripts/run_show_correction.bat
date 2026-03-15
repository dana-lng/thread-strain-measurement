@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Aktuelles Verzeichnis: %CD%
..\..\nn_marker_tracking\venv\Scripts\python.exe show_correction.py
echo.
pause
