param(
    [Parameter(Mandatory = $true)]
    [string]$Experiment
)

$ErrorActionPreference = 'Stop'

$srcDir = Split-Path $PSScriptRoot -Parent
$root = Split-Path $srcDir -Parent
$scriptPath = Join-Path $PSScriptRoot 'rectify_experiment.py'
$pythonExe = 'C:\Users\danal\AppData\Local\Programs\Python\Python311\python.exe'

& $pythonExe $scriptPath $Experiment
