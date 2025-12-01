$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

New-Item -ItemType Directory -Force -Path ".\data"
New-Item -ItemType Directory -Force -Path ".\outputs"
New-Item -ItemType Directory -Force -Path ".\outputs\$timestamp"
New-Item -ItemType Directory -Force -Path ".\outputs\$timestamp\plots"

# Write-Host "Running data preparation..."
# python src/data_preparing.py --timestamp $timestamp

# Write-Host "Running data cleaning..."
# python src/data_cleaning.py --timestamp $timestamp

Write-Host "Running data processing..."
python src/data_processing.py --timestamp $timestamp

Write-Host "Running model training..."
python src/train.py --timestamp $timestamp

Write-Host "Running evaluation..."
python src/evaluation.py --timestamp $timestamp

Write-Host "Pipeline finished successfully."