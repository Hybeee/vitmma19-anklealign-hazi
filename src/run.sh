#! /bin/bash
set -e

timestamp=$(date +%Y%m%d_%H%M%S)

mkdir -p data
mkdir -p outputs

echo "Running data preparation..."
python src/data_pipeline/data_preparing.py --timestamp $timestamp

echo "Running data cleaning..."
python src/data_pipeline/data_cleaning.py --timestamp $timestamp

echo "Running data processing..."
python src/data_pipeline/data_processing.py --timestamp $timestamp

echo "Running model training..."
python src/train.py --timestamp $timestamp

echo "Running evaluation..."
python src/evaluation.py --timestamp $timestamp

echo "Pipeline finished successfully."