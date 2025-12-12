param (
    [switch]$Serve
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

New-Item -ItemType Directory -Force -Path ".\data"
New-Item -ItemType Directory -Force -Path ".\outputs"

Write-Host "Running data preparation..."
python src/data_pipeline/data_preparing.py --timestamp $timestamp --logs_dir_write True

Write-Host "Running data cleaning..."
python src/data_pipeline/data_cleaning.py --timestamp $timestamp

Write-Host "Running data processing..."
python src/data_pipeline/data_processing.py --timestamp $timestamp

Write-Host "Running model training..."
python src/train.py --timestamp $timestamp

Write-Host "Running evaluation..."
python src/evaluation.py --timestamp $timestamp

Write-Host "Running inference..."
python src/inference.py --timestamp $timestamp

if ($Serve) {
    Write-Host "Pipeline finished successfully. Starting MLaaS (API + UI) demo..."

    $apiProcess = Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "api:app", "--app-dir", "src", "--host", "127.0.0.1", "--port", "8000" `
        -PassThru `
        -NoNewWindow

    try {
        Write-Host "Waiting for API to start..."
        Start-Sleep -Seconds 5

        streamlit run src/ui.py
    }
    finally {
        if ($apiProcess -and -not $apiProcess.HasExited) {
            Write-Host "Stopping API server..."
            Stop-Process -Id $apiProcess.Id -Force
        }
    }
}
else {
    Write-Host "Pipeline finished successfully."
}