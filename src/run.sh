#! /bin/bash
set -e

timestamp=$(date +%Y%m%d_%H%M%S)

mkdir -p data
mkdir -p outputs

echo "Running data preparation..."
python data_pipeline/data_preparing.py --timestamp $timestamp

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

echo "Pipeline finished successfully."