#! /bin/bash
set -e

timestamp=$(date +%Y%m%d_%H%M%S)

mkdir -p data
mkdir -p outputs

echo "Running data preparation..."
python data_pipeline/data_preparing.py --timestamp $timestamp --overwrite_logs

echo "Running data cleaning..."
python data_pipeline/data_cleaning.py --timestamp $timestamp

echo "Running data processing..."
python data_pipeline/data_processing.py --timestamp $timestamp

echo "Running model training..."
python train.py --timestamp $timestamp

echo "Running evaluation..."
python evaluation.py --timestamp $timestamp

echo "Running inference..."
python inference.py --timestamp $timestamp

if [[ "$1" == "--serve" ]]; then
    echo "Pipeline finished successfully. Starting MLaaS (API + UI) demo..."

    uvicorn api:app --host 0.0.0.0 --port 8000 &

    API_PID=$!

    trap "kill $API_PID" EXIT

    sleep 5

    streamlit run ui.py --server.address 0.0.0.0 --server.port 8501 --server.headless true

    kill $API_PID
else
    echo "Pipeline finished successfully."
fi