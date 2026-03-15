@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  Schritt 3: Training - Gruppe auswaehlen
echo ============================================================
echo.
echo  Gruppe A: Versuch 1+2  (1/6060 s, 2000 fps)  weisser Hintergrund
echo  Gruppe B: Versuch 3-6  (1/8000 s, 2000 fps)  Fadenstruktur sichtbar
echo  Gruppe C: Versuch 8-11 (1/12500 s, 6000 fps)  Fadenstruktur stark
echo.
echo  Ausgabe: checkpoints\Group_X\best_model.pt
echo.

REM ============================================================
REM  HIER ANPASSEN: A, B oder C eintragen
REM ============================================================
SET GRUPPE=A

REM ============================================================
REM  Gruppe A (Versuch_1 + Versuch_2)
REM ============================================================
IF "%GRUPPE%"=="A" (
    echo [Gruppe A] Versuch_1 + Versuch_2  ^(587 Annotationen^)
    ..\..\nn_marker_tracking\venv\Scripts\python.exe 3_train.py ^
        --annotations ^
            ..\Versuch_1\annotations.json ^
            ..\Versuch_2\annotations.json ^
        --images ^
            ..\Versuch_1 ^
            ..\Versuch_2 ^
        --calib   camera_calibration.json ^
        --model   standard ^
        --base_ch 16 ^
        --epochs  150 ^
        --batch   16 ^
        --lr      1e-3 ^
        --output  checkpoints\Group_A
    goto done
)

REM ============================================================
REM  Gruppe B (Versuch_3 bis Versuch_6)
REM ============================================================
IF "%GRUPPE%"=="B" (
    echo [Gruppe B] Versuch_3 bis Versuch_6
    ..\..\nn_marker_tracking\venv\Scripts\python.exe 3_train.py ^
        --annotations ^
            ..\Versuch_3\annotations.json ^
            ..\Versuch_4\annotations.json ^
            ..\Versuch_5\annotations.json ^
            ..\Versuch_6\annotations.json ^
        --images ^
            ..\Versuch_3 ^
            ..\Versuch_4 ^
            ..\Versuch_5 ^
            ..\Versuch_6 ^
        --calib   camera_calibration.json ^
        --model   standard ^
        --epochs  150 ^
        --batch   16 ^
        --lr      1e-3 ^
        --output  checkpoints\Group_B
    goto done
)

REM ============================================================
REM  Gruppe C (Versuch_8 bis Versuch_11)
REM ============================================================
IF "%GRUPPE%"=="C" (
    echo [Gruppe C] Versuch_8 bis Versuch_11
    ..\..\nn_marker_tracking\venv\Scripts\python.exe 3_train.py ^
        --annotations ^
            ..\Versuch_8\annotations.json ^
            ..\Versuch_9\annotations.json ^
            ..\Versuch_10\annotations.json ^
            ..\Versuch_11\annotations.json ^
        --images ^
            ..\Versuch_8 ^
            ..\Versuch_9 ^
            ..\Versuch_10 ^
            ..\Versuch_11 ^
        --calib   camera_calibration.json ^
        --model   standard ^
        --epochs  150 ^
        --batch   16 ^
        --lr      1e-3 ^
        --output  checkpoints\Group_C
    goto done
)

echo FEHLER: Unbekannte Gruppe "%GRUPPE%". Bitte A, B oder C eintragen.
pause
exit /b 1

:done
echo.
if %ERRORLEVEL% EQU 0 (
    echo ERFOLG: Bestes Modell in checkpoints\Group_%GRUPPE%\best_model.pt
    echo         Lernkurven: checkpoints\Group_%GRUPPE%\learning_curves.png
) else (
    echo FEHLER: Training fehlgeschlagen.
    echo Tipp: Annotationsdateien pruefen - muessen alle vorhanden sein.
)
pause
