param(
    [Parameter(Mandatory = $true)]
    [string]$Experiment
)

$ErrorActionPreference = 'Stop'

$srcDir = Split-Path $PSScriptRoot -Parent
$root = Split-Path $srcDir -Parent
$inputDir = Join-Path $root "data\raw\$Experiment"
$scriptPath = Join-Path $PSScriptRoot "rectify_first_versuch1.py"
$pythonExe = "C:\Users\danal\AppData\Local\Programs\Python\Python311\python.exe"

if (-not (Test-Path $inputDir)) {
    throw "Experiment nicht gefunden: $inputDir"
}

Get-ChildItem -Path $inputDir -Filter *.tif |
    Sort-Object Name |
    ForEach-Object {
        & $pythonExe $scriptPath $_.FullName
    }
